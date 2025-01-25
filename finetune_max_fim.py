from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import torch_save, train_diag_fim_logtr
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

        # Move model to device
        device = args.device
        model = model.to(device)

        # Define optimizer and loss function
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = nn.CrossEntropyLoss()

        # Variables for tracking the best model based on FIM log-trace
        best_fim_logtrace = float("-inf")
        best_model = None

        # Fine-tune the model
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

            # Compute the log-trace of the diagonal Fisher Information Matrix
            samples_nr = 2000  # Number of per-example gradients to accumulate
            fim_logtrace = train_diag_fim_logtr(args, model, f"{dataset_name}Val", samples_nr)
            print(f"Epoch {epoch + 1}/{num_epochs}, FIM Log-Trace: {fim_logtrace:.4f}")

            # Update the best model based on FIM log-trace
            if fim_logtrace > best_fim_logtrace:
                best_fim_logtrace = fim_logtrace
                best_model = encoder

        # Save the best fine-tuned encoder
        save_path = os.path.join(args.save, f"{dataset_name}_encoder_ft.pt")
        if best_model is not None:
            encoder.save(save_path)
            print(f"Saved fine-tuned encoder for {dataset_name} at {save_path} (selected by FIM log-trace)")

if __name__ == "__main__":
    args = parse_arguments()
    finetune_model(args)
