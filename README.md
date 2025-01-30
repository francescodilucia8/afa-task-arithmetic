# AML & DAAI 2024/2025 Project - Optimizing Task Arithmetic: Multi-Task Learning and Complexity Trade-offs Across Training Conditions
Official repository for the "__Optimizing Task Arithmetic: Multi-Task Learning and Complexity Trade-offs Across Training Conditions__" project - Advanced Machine Learning & Data Analysis and Artificial Intelligence Courses 2024/2025 @ PoliTo

## Getting Started
Make sure to have a CUDA capable device, supporting at least CUDA 11.8, installed and correctly configured on your system. 

(The base code of this project has been produced using CUDA 11.8 and Python 3.10.9)

Follow https://pytorch.org/get-started/locally/ to setup PyTorch (note that PyTorch comes pre-installed on Google Colab's environments, so you can skip this step)

Once you have properly setup everything, make sure you are in the correct directory (using your path):
```bash
%cd .../polito-task-arithmetic
```
Run from the command line:
```bash
pip install -r requirements.txt
```


### Dataset
Download to your disk/drive the datasets from the provided drive folder (see project report). Then, unzip them to some location.
To avoid unzipping manually it is suggested to execute the following commands (making sure you are using the right path):
```bash
!apt install unzip
```
```bash
!unzip ".../datasets/*" -d ".../data/"
```
After this set of commands it is finally possible to use the commands found in the __base.sh__ file to perform the experiments.

## Base Code Structure
In the following, you can find a brief description of the included files.

| File/Folder | Description |
| ---- | ----------- |
| `args.py` | contains the function responsible for parsing each command line argument. |
| `datasets/` | contains the files with code to load data, build splits and dataloaders. |
| `utils.py` | contains several utilities to correctly setup the pipeline. |
| `task_vectors.py` | contains the code for building task vectors and/or load checkpoints. |
| `modeling.py` | contains the backbone architectures and modules used in the project. |
| `heads.py` | contains the logic to build and store the open-vocabulary classifiers used in the project. |
| `finetune.py` | contains the logic to fine-tune the pre-trained model on each of the six provided datasets and save each finetuned checkpoint resulting from the completion of the last epoch. |
| `finetune_max_fim.py` | contains the logic to fine-tune the pre-trained model on each of the six provided datasets and for each one save only the finetuned checkpoint characterized by maximal log-trace diagonal FIM. |
| `finetune_max_val.py` | contains the logic to fine-tune the pre-trained model on each of the six provided datasets and for each one save only the finetuned checkpoint characterized by highest validation accuracy . |
| `eval_single_task.py` | contains the logic to obtain and save the train and test absolute accuracies (one accuracy per task) and their complexity |
| `eval_task_addition.py` | contains the logic to obtain and save the average normalized accuracy of the model obtained through task addition on validation split (while also computing the best scaling factor α) and its complexity. |
| `eval_scaled_single_task.py` | contains the logic to obtain and save the average absolute accuracy and average normalized accuracy on test splits of the six tasks (computed using the best scaling factor α) and their complexity. |

## Running The Experiments
To run the experiments you can use, copy and modify the provided launch script `launch_scripts/base.sh`. The file does not contain a specific order to execute experiments but provides all the commands and placeholders for command line arguments and can be modified to run and reproduce specific experiments if needed.

### Basic Command Line Arguments
| Argument &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;  | Description |
| -------- | ----------- |
| `--model` | the name of the architecture used in the experiments (keep it as `ViT-B-32`) |
| `--batch-size` | batch size used in the optimization procedure (default: `32`) |
| `--lr` | learning rate used in the optimization procedure (default: `1e-4`) |
| `--wd` | weight decay of the optimizer (default: `0.0`) |
| `--data-location` | path to the folder containing your unzipped dataset folders. |
| `--save` | path to the folder where to save your results. |
| `--eval-datasets` | datasets used for evaluation. |
| `--alpha` | scaling factor α used in the evaluation of scaled single task. |

### Useful information to run experiments on balanced train datasets

In the folder `datasets` the file "**registry.py**" has been modified in order to provide the necessary operations for dataset balancing .

The following has been added/modified : 
- added the function "`balance_classes`"
- modified "`split_train_into_train_val`" to support the new function for balancing : instead of using the base dataset , it calls the balancing function which returns the balanced dataset used then to produce the splits.

**Remark** : "registry.py" (in the current state in the repository) contains the code for balancing but such code is commented while the original code is uncomment and ready to use for the experiments without balancing. In order to execute experiments using balanced datasets it is then necessary to comment the base version of "`split_train_into_train_val`" function and uncomment the block of code added for the balanced scenarios.