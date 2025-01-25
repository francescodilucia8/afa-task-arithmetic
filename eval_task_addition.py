import os
import json
import numpy as np
from tqdm import tqdm
from args import parse_arguments
from task_vectors import NonLinearTaskVector
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
import torch
from torch import nn
from utils import torch_save, torch_load, train_diag_fim_logtr

def eval_task_addition(args):
    task_vectors = []
    single_task_accuracies = {}  # To store fine-tuned single-task accuracies for normalization
    datasets_cache = {}  # Cache to store datasets and dataloaders for reuse

    # Load task vectors for all datasets
    for dataset_name in args.eval_datasets:
        pretrained_path = os.path.join(args.save, "pretrained_encoder.pt")
        finetuned_path = os.path.join(args.save, f"{dataset_name}_encoder_ft.pt")
        task_vector = NonLinearTaskVector(pretrained_path, finetuned_path)
        task_vectors.append(task_vector)

        # Load dataset and dataloader once and cache them
        train_split_name = f"{dataset_name}Val"
        test_split_name = dataset_name

        # Compute single-task fine-tuned accuracy
        head = get_classification_head(args, train_split_name)
        single_task_model = ImageClassifier(task_vector.apply_to(pretrained_path, scaling_coef=1.0), head).to(args.device)
        single_task_model.eval()


        if dataset_name not in datasets_cache:
            datasets_cache[dataset_name] = {
                "val": get_dataloader(
                    get_dataset(
                        train_split_name,
                        preprocess=single_task_model.val_preprocess,  
                        location=args.data_location,
                        batch_size=args.batch_size,
                        num_workers=2
                    ),
                    is_train=False,
                    args=args
                ),
                "train": get_dataloader(
                    get_dataset(
                        train_split_name,
                        preprocess=single_task_model.val_preprocess,  
                        location=args.data_location,
                        batch_size=args.batch_size,
                        num_workers=2
                    ),
                    is_train=True,
                    args=args
                ),
                "test": get_dataloader(
                    get_dataset(
                        test_split_name,
                        preprocess=single_task_model.val_preprocess,
                        location=args.data_location,
                        batch_size=args.batch_size,
                        num_workers=2
                    ),
                    is_train=False,
                    args=args
                )
            }


        # Initialize the dictionary for the current dataset
        single_task_accuracies[dataset_name] = {}

        for split in ["val", "train", "test"]:
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(datasets_cache[dataset_name][split], desc=f"Evaluating single-task accuracy for {dataset_name} on {split}"):
                    data = maybe_dictionarize(batch)
                    images, labels = data["images"].to(args.device), data["labels"].to(args.device)
                    outputs = single_task_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            single_task_accuracy = correct / total
            # Save the accuracy for the current split
            single_task_accuracies[dataset_name][split] = single_task_accuracy
            print(f"Single-task accuracy for {dataset_name} on {split}: {single_task_accuracy:.4f}")

    # Search for the best scaling coefficient alpha
    best_alpha = None
    best_normalized_accuracy = 0
    alphas = np.arange(0.0, 1.05, 0.05)
    multi_task_accuracies = {}  # Store multi-task accuracies for each alpha

    for alpha in alphas:
        print(f"Evaluating with alpha = {alpha:.2f}")
        combined_task_vector = sum(tv * alpha for tv in task_vectors)
        combined_encoder = combined_task_vector.apply_to(pretrained_path)

        # Evaluate the combined model on all datasets
        total_absolute_accuracy = 0
        total_normalized_accuracy = 0
        for dataset_name in args.eval_datasets:
            head = get_classification_head(args, f"{dataset_name}Val")
            multi_task_model = ImageClassifier(combined_encoder, head).to(args.device)
            multi_task_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(datasets_cache[dataset_name]["val"], desc=f"Evaluating multi-task model for {dataset_name} on validation split"):
                    data = maybe_dictionarize(batch)
                    images, labels = data["images"].to(args.device), data["labels"].to(args.device)
                    outputs = multi_task_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            multi_task_accuracy = correct / total
            print(f"Multi-task accuracy with alpha={alpha:.2f} for {dataset_name}: {multi_task_accuracy:.4f}")
            total_absolute_accuracy += multi_task_accuracy

            # Normalized accuracy
            single_task_accuracy = single_task_accuracies[dataset_name]["val"]
            total_normalized_accuracy += multi_task_accuracy / single_task_accuracy

        avg_absolute_accuracy = total_absolute_accuracy / len(args.eval_datasets)
        avg_normalized_accuracy = total_normalized_accuracy / len(args.eval_datasets)

        multi_task_accuracies[alpha] = {
            "absolute": avg_absolute_accuracy,
            "normalized": avg_normalized_accuracy
        }

        if avg_normalized_accuracy > best_normalized_accuracy:
            best_alpha = alpha
            best_normalized_accuracy = avg_normalized_accuracy

    print(f"Best alpha: {best_alpha:.2f}, Best normalized accuracy on validation split: {best_normalized_accuracy:.4f}")

    # Evaluate the best multi-task model on the train and test splits
    best_combined_task_vector = sum(tv * best_alpha for tv in task_vectors)
    best_combined_encoder = best_combined_task_vector.apply_to(pretrained_path)
    final_results = {}
    final_results["best_alpha"] = best_alpha

    # Initialize totals for averages
    total_absolute_accuracy_train = 0
    total_normalized_accuracy_train = 0
    total_absolute_accuracy_test = 0
    total_normalized_accuracy_test = 0

    for dataset_name in args.eval_datasets:
        head = get_classification_head(args, dataset_name)
        multi_task_model = ImageClassifier(best_combined_encoder, head).to(args.device)
        multi_task_model.eval()

        for split in ["Train", "Test"]:
            dataloader = datasets_cache[dataset_name]["train"] if split == "Train" else datasets_cache[dataset_name]["test"]

            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Evaluating best multi-task model for {dataset_name} on {split}"):
                    data = maybe_dictionarize(batch)
                    images, labels = data["images"].to(args.device), data["labels"].to(args.device)
                    outputs = multi_task_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            multi_task_accuracy = correct / total

            # Save per-split accuracies
            if split == "Train":
                total_absolute_accuracy_train += multi_task_accuracy
                total_normalized_accuracy_train += multi_task_accuracy / single_task_accuracies[dataset_name]["train"]


                final_results[f"{dataset_name}_train"] = {
                    "absolute_accuracy": multi_task_accuracy,
                    "normalized_accuracy": multi_task_accuracy / single_task_accuracies[dataset_name]["train"]
                }
            else:  # Test split
                total_absolute_accuracy_test += multi_task_accuracy
                total_normalized_accuracy_test += multi_task_accuracy /single_task_accuracies[dataset_name]["test"]

                final_results[f"{dataset_name}_test"] = {
                    "absolute_accuracy": multi_task_accuracy,
                    "normalized_accuracy": multi_task_accuracy / single_task_accuracies[dataset_name]["test"]
                }
        # Compute the log-trace of the Fisher diagonal for the model (on Train split)
        samples_nr = 2000  # Number of per-example gradients to accumulate
        logdet_hF = train_diag_fim_logtr(args, multi_task_model, f"{dataset_name}Val", samples_nr)
        final_results[f"{dataset_name}_multitask_logtrace_fisher"] = logdet_hF
        print(f"{dataset_name} - Log-Trace of Fisher Diagonal : {logdet_hF:.4f}")

    # Compute averages for train and test splits
    final_results["avg_train_absolute_accuracy"] = total_absolute_accuracy_train / len(args.eval_datasets)
    final_results["avg_train_normalized_accuracy"] = total_normalized_accuracy_train / len(args.eval_datasets)
    final_results["avg_test_absolute_accuracy"] = total_absolute_accuracy_test / len(args.eval_datasets)
    final_results["avg_test_normalized_accuracy"] = total_normalized_accuracy_test / len(args.eval_datasets)

    # Save results
    output_path = os.path.join(args.save, "task_addition_results.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    args = parse_arguments()
    eval_task_addition(args)
