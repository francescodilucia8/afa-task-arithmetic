from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import torch_save
import torch
from torch import nn, optim
import os

# Define the number of epochs per dataset
epochs_per_dataset = {
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SVHN": 4,
}

def finetune_model(args):
    for dataset_name, num_epochs in epochs_per_dataset.items():
        print(f"Starting fine-tuning for {dataset_name}...")

        # Instantiate pre-trained encoder and classification head
        encoder = ImageEncoder(args)
        # Save pre-trained encoder
        save_path0 = os.path.join(args.save, "pretrained_encoder.pt")
        # Check if the pre-trained encoder already exists
        if not os.path.exists(save_path0):
            print(f"Saving pre-trained encoder to {save_path0}...")
            encoder.save(save_path0)

        head = get_classification_head(args, f"{dataset_name}Val")
        model = ImageClassifier(encoder, head)
        model.freeze_head()  # Freeze the classification head

        # Obtain train dataset and loader
        train_dataset = get_dataset(
            f"{dataset_name}Val",
            preprocess=model.train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=2
        )
        train_loader = get_dataloader(train_dataset, is_train=True, args=args)

        # Obtain validation dataset and loader
        val_dataset = get_dataset(
            f"{dataset_name}Val",
            preprocess=model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=2
        )
        val_loader = get_dataloader(val_dataset, is_train=False, args=args)

        # Move model to device
        device = args.device
        model = model.to(device)

        # Define optimizer and loss function
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = nn.CrossEntropyLoss()

        # Fine-tune the model
        best_val_accuracy = 0.0
        best_model = None

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                data = maybe_dictionarize(batch)
                images, labels = data["images"].to(device), data["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Evaluate on validation set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    data = maybe_dictionarize(batch)
                    images, labels = data["images"].to(device), data["labels"].to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            val_accuracy = correct / total
            print(f"Validation Accuracy after Epoch {epoch + 1}: {val_accuracy:.4f}")

            # Save the model with the best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = encoder

            model.train()

        # Save the fine-tuned encoder with the best validation accuracy
        if best_model is not None:
            save_path = os.path.join(args.save, f"{dataset_name}_encoder_ft.pt")
            best_model.save(save_path)
            print(f"Saved fine-tuned encoder for {dataset_name} with best validation accuracy at {save_path}")

if __name__ == "__main__":
    args = parse_arguments()
    finetune_model(args)
