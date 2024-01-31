#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J runname
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
#module load gcc/10.3.0-binutils-2.36.1
#module load gcc/12.3.0-binutils-2.40
#module load sqlite3/3.42.0
#module load python3/3.11.4
#module load cython/0.29.35-python-3.11.4
#module load cuda/11.6

source ../myVirtEnv/bin/activate

/appl/python/3.11.4/bin/python3.11 EL_train.py --save_model_and_optim_epochs 2 --run_id 0 --continue_trainings 0 --epoch_continue_from 0 --num_epochs 25 --data_imbalance_handler 1 --evaluate_every_epochs 2 --activate_custom_epoch 1 --custom_len_epoch 1000 --model_name FasterRCNN_ResNet-50-FPN --horizontal_flip 1 --vertical_flip 1 --adjust_sharpness 1 --random_equalize 1 --random_gamma 1 --gaussian_blur 1 --color_jitter 1 --cut_paste_learn 1 --trainable_backbone_layers 3 --learning_rate 0.0003 --scheduler_gamma 0.995 --weight_decay 1e-5 --label_smoothing 0.05 --experiment_name example_experiment_0 --run_name run_0

deactivate