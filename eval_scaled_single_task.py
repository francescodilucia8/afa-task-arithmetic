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


def eval_scaled_single_task(args):
   
    results = {}  # To store all the scaled fine-tuned results

    # Load task vectors for all datasets
    for dataset_name in args.eval_datasets:
        pretrained_path = os.path.join(args.save, "pretrained_encoder.pt")
        finetuned_path = os.path.join(args.save, f"{dataset_name}_encoder_ft.pt")
        task_vector = NonLinearTaskVector(pretrained_path, finetuned_path)

        # Compute scaled single-task fine-tuned accuracy
        head = get_classification_head(args, f"{dataset_name}Val")
        single_task_model = ImageClassifier(task_vector.apply_to(pretrained_path, scaling_coef=args.alpha), head).to(args.device)
        single_task_model.eval()
        for split in ["Train", "Test"]:
            split_name = f"{dataset_name}Val" if split == "Train" else dataset_name
            dataset = get_dataset(
                split_name,
                preprocess=single_task_model.val_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                num_workers=2
            )
            dataloader = get_dataloader(dataset, is_train=True if split == "Train" else False, args=args)

            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Evaluating scaled(alpha=0.3) single-task accuracy for {dataset_name} on {split}"):
                    data = maybe_dictionarize(batch)
                    images, labels = data["images"].to(args.device), data["labels"].to(args.device)
                    outputs = single_task_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            single_task_accuracy = correct / total
            results[f"{dataset_name}_scaled_finetuned_{split.lower()}_accuracy"] = single_task_accuracy
            print(f" Scaled single-task accuracy for {dataset_name} on {split}: {single_task_accuracy:.4f}")
        
        # Compute the log-trace of the Fisher diagonal for the model (on Train split)
        samples_nr = 2000  # Number of per-example gradients to accumulate
        logdet_hF = train_diag_fim_logtr(args, single_task_model, f"{dataset_name}Val", samples_nr)
        results[f"{dataset_name}_finetuned_logtrace_fisher"] = logdet_hF
        print(f"{dataset_name} - Log-Trace of Fisher Diagonal : {logdet_hF:.4f}")
    # Save the results to the JSON file
        save_path = os.path.join(args.save, "scaled_single_task_eval_results.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Saved evaluation results to {save_path}")

if __name__ == "__main__":
    args = parse_arguments()

    # Ensure evaluation datasets are provided
    if not args.eval_datasets:
        raise ValueError("No evaluation datasets specified. Use the --eval-datasets argument.")

    eval_scaled_single_task(args)