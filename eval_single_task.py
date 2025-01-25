import json
import os
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import torch_save, torch_load, train_diag_fim_logtr
import torch
from torch import nn
from tqdm.auto import tqdm

def evaluate_model(args):
    results = {}

    for dataset_name in args.eval_datasets:
        print(f"Evaluating model for {dataset_name}...")

        # Paths for pre-trained and fine-tuned encoders
        pretrained_path = os.path.join(args.save, "pretrained_encoder.pt")
        finetuned_path = os.path.join(args.save, f"{dataset_name}_encoder_ft.pt")

         # Load the pre-trained encoder
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pre-trained encoder not found at {pretrained_path}")
        pretrained_encoder = torch_load(pretrained_path, args.device)

        # Load the fine-tuned encoder
        if not os.path.exists(finetuned_path):
            raise FileNotFoundError(f"Fine-tuned encoder not found for {dataset_name} at {finetuned_path}")
        finetuned_encoder = torch_load(finetuned_path, args.device)

        # Load the classification head
        head = get_classification_head(args, f"{dataset_name}Val")

        # Build the models
        pretrained_model = ImageClassifier(pretrained_encoder, head).to(args.device)
        finetuned_model = ImageClassifier(finetuned_encoder, head).to(args.device)

         # Set models to evaluation mode
        pretrained_model.eval()
        finetuned_model.eval()

        # Initialize loss function
        criterion = nn.CrossEntropyLoss()

        # Evaluate both models on Train and test splits 
        for model_name, model in zip(["finetuned"], [finetuned_model]):
            for split in ["Train", "Test"]:
                split_name = f"{dataset_name}Val" if split == "Train" else dataset_name

                dataset = get_dataset(
                    split_name,
                    preprocess=model.val_preprocess,
                    location=args.data_location,
                    batch_size=args.batch_size,
                    num_workers=2
                )

                dataloader = get_dataloader(dataset,  is_train=True if split == "Train" else False, args=args)

                total_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch in tqdm(dataloader, desc=f"Evaluating {model_name} model on {split} Split"):
                        data = maybe_dictionarize(batch)
                        images, labels = data["images"].to(args.device), data["labels"].to(args.device)

                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        total_loss += loss.item() * labels.size(0)  # Multiply by batch size
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(labels).sum().item()
                        total += labels.size(0)

                avg_loss = total_loss / total
                accuracy = correct / total

                print(f"{model_name.capitalize()} Model - {split} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

                # Save results
                #results[f"{dataset_name}_{model_name}_{split.lower()}_loss"] = avg_loss
                results[f"{dataset_name}_{model_name}_{split.lower()}_accuracy"] = accuracy

            # Compute the log-trace of the Fisher diagonal for the model (on Train split)
            samples_nr = 2000  # Number of per-example gradients to accumulate
            logdet_hF = train_diag_fim_logtr(args, model, f"{dataset_name}Val", samples_nr)
            results[f"{dataset_name}_{model_name}_logtrace_fisher"] = logdet_hF
            print(f"{dataset_name} - Log-Trace of Fisher Diagonal ({model_name.capitalize()}): {logdet_hF:.4f}")

        # Save the results to the JSON file
        save_path = os.path.join(args.save, "single_task_eval_results.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Saved evaluation results to {save_path}")

if __name__ == "__main__":
    args = parse_arguments()

    # Ensure evaluation datasets are provided
    if not args.eval_datasets:
        raise ValueError("No evaluation datasets specified. Use the --eval-datasets argument.")

    evaluate_model(args)
