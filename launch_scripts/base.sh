

# finetune on all datasets saving as checkpoint the version obtained completing the last epoch
python finetune.py \
--data-location=/path/to/datasets/ \
--save=/path/to/save/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0 \
--model="ViT-B-32-quickgelu"

# finetune on all datasets saving as checkpoint the version of each characterized by maximal log-trace diagonal FIM.
python finetune_max_fim.py \
--data-location=/path/to/datasets/ \
--save=/path/to/save/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0 \
--model="ViT-B-32-quickgelu"

# finetune on all datasets saving as checkpoint the version of each characterized by highest validation accuracy
python finetune_max_val.py \
--data-location=/path/to/datasets/ \
--save=/path/to/save/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0 \
--model="ViT-B-32-quickgelu"

# eval single task
python eval_single_task.py \
--eval-datasets="DTD,EuroSAT,GTSRB,MNIST,RESISC45,SVHN" \
--data-location=/path/to/datasets/ \
--save=/path/to/save/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0 \
--model="ViT-B-32-quickgelu"

# eval task addition
python eval_task_addition.py \
--eval-datasets="DTD,EuroSAT,GTSRB,MNIST,RESISC45,SVHN" \
--data-location=/path/to/datasets/ \
--save=/path/to/save/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0 \
--model="ViT-B-32-quickgelu"

# eval scaled single task
python eval_scaled_single_task.py \
--eval-datasets="DTD,EuroSAT,GTSRB,MNIST,RESISC45,SVHN" \
--data-location=/path/to/datasets/ \
--save=/path/to/save/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0 \
--model="ViT-B-32-quickgelu" \
--alpha=0.3  # Example value, replace with the best alpha

