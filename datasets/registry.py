import sys
import inspect
import random
import torch
import copy
from collections import defaultdict

from torch.utils.data.dataset import random_split

from datasets.cars import Cars
from datasets.cifar10 import CIFAR10
from datasets.cifar100 import CIFAR100
from datasets.dtd import DTD

from datasets.eurosat import EuroSAT, EuroSATVal # comment this to force the balance operation on the train split avoiding the direct use of the predefined splits 
#from datasets.eurosat import EuroSAT			# and uncomment this instead
from datasets.gtsrb import GTSRB
from datasets.imagenet import ImageNet
from datasets.mnist import MNIST
from datasets.resisc45 import RESISC45
from datasets.stl10 import STL10
from datasets.svhn import SVHN
from datasets.sun397 import SUN397

registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None

# Comment this function and uncomment the block below for balancing
def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )
    if new_dataset_class_name == 'MNISTVal':
        assert trainset.indices[0] == 36044


    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset

################################################# UNCOMMENT THIS BLOCK FOR BALANCING ( and comment the standard split_train_into_train_val ) ##########################################
# def balance_classes(dataset,dataset_name):
    # from collections import Counter  # For easy class counting
    # print("DATASET NAME : " , dataset_name)
    # # Handle the specific "TEST" dataset case
    # if dataset_name == "RESISC45Val":
        # print("Detected 'RESISC45' dataset: Converting tensor labels to integers.")
        # # Wrap the dataset to handle tensor labels
        # dataset = [(data, label.item() if torch.is_tensor(label) else label) for data, label in dataset]

    # # Group indices by class
    # class_to_indices = defaultdict(list)
    # for idx, (_, label) in enumerate(dataset):
        # class_to_indices[label].append(idx)
    
    # # Print original class distribution
    # original_class_sizes = {cls: len(indices) for cls, indices in class_to_indices.items()}
    # print("Original class distribution:", original_class_sizes)
    
    # # Determine the minimum class size
    # min_class_size = min(len(indices) for indices in class_to_indices.values())
    # print("Minimum class size (target for balancing):", min_class_size)
    
    # # Subsample each class to the minimum size
    # balanced_indices = []
    # for indices in class_to_indices.values():
        # balanced_indices.extend(random.sample(indices, min_class_size))
    
    # # Create a balanced subset
    # balanced_dataset = torch.utils.data.Subset(dataset, balanced_indices)
    
    # # Count and print the new class distribution
    # balanced_class_counts = Counter(dataset[idx][1] for idx in balanced_indices)
    # print("Balanced class distribution:", dict(balanced_class_counts))
    
    # return balanced_dataset

# def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    # assert val_fraction > 0. and val_fraction < 1.

    # # Balance the training dataset
    # balanced_train_dataset = balance_classes(dataset.train_dataset,new_dataset_class_name)

    # # Calculate sizes based on the balanced dataset
    # total_size = len(balanced_train_dataset)
    # val_size = int(total_size * val_fraction)
    # if max_val_samples is not None:
        # val_size = min(val_size, max_val_samples)
    # train_size = total_size - val_size

    # # Ensure sizes are valid
    # assert val_size > 0, "Validation size must be greater than 0."
    # assert train_size > 0, "Training size must be greater than 0."

    # # Perform the split
    # lengths = [train_size, val_size]
    # trainset, valset = random_split(
        # balanced_train_dataset,
        # lengths,
        # generator=torch.Generator().manual_seed(seed)
    # )

    # # Add print statements to inspect train and val splits
    # from collections import Counter  # Import for counting labels
    # train_labels = [balanced_train_dataset[idx][1] for idx in trainset.indices]
    # val_labels = [balanced_train_dataset[idx][1] for idx in valset.indices]
    # print(f"Train split label distribution: {dict(Counter(train_labels))}")
    # print(f"Validation split label distribution: {dict(Counter(val_labels))}")

    # # If needed for debugging (specific to MNISTVal)
    # if new_dataset_class_name == 'MNISTVal':
        # assert trainset.indices[0] == 31034

    # # Create a new dataset object
    # new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    # new_dataset = new_dataset_class()

    # # Assign datasets and create data loaders
    # new_dataset.train_dataset = trainset
    # new_dataset.train_loader = torch.utils.data.DataLoader(
        # new_dataset.train_dataset,
        # shuffle=True,
        # batch_size=batch_size,
        # num_workers=num_workers,
    # )

    # new_dataset.test_dataset = valset
    # new_dataset.test_loader = torch.utils.data.DataLoader(
        # new_dataset.test_dataset,
        # batch_size=batch_size,
        # num_workers=num_workers,
    # )

    # new_dataset.classnames = copy.copy(dataset.classnames)

    # return new_dataset
####################################################################################################################################################

def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=2, val_fraction=0.1, max_val_samples=5000):
    if dataset_name.endswith('Val'):
        # Handle val splits
        if dataset_name in registry:
            print("Specified dataset is in the registry")
            print("Getting the validation split dataset...")
            dataset_class = registry[dataset_name]
        else:
            print("Specified dataset is not in registry...")
            print("Starting split_train_into_train_val procedure...")
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        print("Getting the base training dataset...")
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers
    )
    return dataset
