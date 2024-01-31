import torch  
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler 

import numpy as np
import pandas as pd

import random
import time
from datetime import datetime

from EL_models import Model
from EL_dataset import PVDefectsDS, get_transform, collate_fn, Sampler
from EL_train_utils import train_one_epoch
from EL_utils import int_to_boolean
from EL_optim import Optimizer
from EL_validation import evaluate_engine, loss_one_epoch_val, compute_confusion_matrix 
from EL_metrics import Writer

import argparse

import os

import shutil

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

parser = argparse.ArgumentParser(description="EL training",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Reproducibility and device configuration
parser.add_argument("-D", "--deterministic", action='store_true', help="deterministic (slower)")
parser.add_argument("-P", "--pin_memory", action='store_true', help="pin memory in data loader")
parser.add_argument("--seed", type=int, default=0, help="random seed")

# Saving and continue trainings
parser.add_argument("--experiment_name", type=str, default="no_default", help="name of the experiments being performed")
parser.add_argument("--run_name", type=str, default="default", help="name of the specific run experiment, don't repeat a name already used if you dont continue that experiment")
parser.add_argument("-E", "--save_model_and_optim_epochs", type=int, default=2, help="save the model and optim every this number of epochs")
parser.add_argument("--run_id", type=str, default='no_id', help="WandB run id")
parser.add_argument("--continue_trainings", type=int, default=0, help="0 new trainings, 1 continue, need to provide run_id")
parser.add_argument("--epoch_continue_from", type = int, default= 0, help="continue from epoch number X")
parser.add_argument("--load_model_name", type=str, default="no_default", help="name of the model to load")
parser.add_argument("--load_optim_name", type=str, default="no_default", help="name of the optim to load")
parser.add_argument("--load_lr_name", type=str, default="no_default", help="name of the lr scheduler to load")

# Training param
parser.add_argument("--evaluate_every_epochs", type=int, default=5, help="evaluate every X epochs")
parser.add_argument("-e", "--num_epochs", type=int, default=2, help="number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=4, help="batch size")
parser.add_argument("-W", "--num_workers", type=int, default=4, help="number of workers in data loader")

# Optimizer hyper-param
parser.add_argument("--optim_name", type=str, default='Adam')
parser.add_argument("--optim_default", type=int, default=0, help="0 is No default, 1 is default")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-r ", "--rho", type=float, default=0.9)
parser.add_argument("-w", "--weight_decay", type=float, default=1e-5, help="weight decay (L2 regularization)")
parser.add_argument("-m ", "--momentum", type=float, default=0.9, help="momentum of the learning")
parser.add_argument("--dampening", type=float, default=0)
parser.add_argument("--nesterov", type=int, default=0, help="0 is off, 1 is on")
parser.add_argument("-g", "--scheduler_gamma", type=float, default=0.99, help="gamma value for exp decay learning rate scheduler")

# Data augmentation
parser.add_argument("--gaussian_blur", type=int, default=0, help="gaussian_blur, 0 is off, 1 is on")
parser.add_argument("--color_jitter", type=int, default=0, help="color_jitter, 0 is off, 1 is on")
parser.add_argument("--horizontal_flip", type=int, default=0, help="horizontal_flip, 0 is off, 1 is on")
parser.add_argument("--vertical_flip", type=int, default=0, help="vertical_flip, 0 is off, 1 is on")
parser.add_argument("--adjust_sharpness", type=int, default=0, help="adjust_sharpness, 0 is off, 1 is on")
parser.add_argument("--random_gamma", type=int, default=0, help="random_gamma, 0 is off, 1 is on")
parser.add_argument("--gaussian_noise", type=int, default=0, help="gaussian_noise, 0 is off, 1 is on")
parser.add_argument("--random_erasing", type=int, default=0, help="random_erasing, 0 is off, 1 is on")
parser.add_argument("--random_equalize", type=int, default=0, help="random_equalize, 0 is off, 1 is on")
parser.add_argument("--autocontrast", type=int, default=0, help="autocontrast, 0 is off, 1 is on")
parser.add_argument("--cut_paste_learn", type=int, default=0, help="cut_paste_learn, 0 is off, 1 is on")

# Data imbalance
parser.add_argument("--data_imbalance_handler", type=int, default=1, help="weighted sampling mechanism, 0 is off, 1 is on")

# Model
parser.add_argument("--model_name", type=str, default='FasterRCNN_ResNet-50-FPN')
parser.add_argument("--trainable_backbone_layers", type=int, default=3)
parser.add_argument("--label_smoothing", type=float, default=0, help="label_smoothing is not implemented via parser, just via modifying the code")
parser.add_argument("--gamma", type=float, default=2, help="For RetinaNet Focal Loss. Not implemented.")

# Customization
parser.add_argument("--activate_custom_epoch", type=int, default=0, help="activate the modification of the numb of interations per epoch")
parser.add_argument("--custom_len_epoch", type=int, default=0, help="modify the numb of interations per epoch")

args = parser.parse_args()
config = vars(args)
print(config)

#%%
# Reproducibility
seed = config['seed']
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if config['deterministic']:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# New params
experiment_name = str(config['experiment_name'])
run_name = str(config['run_name'])
evaluate_every_epochs = config['evaluate_every_epochs']
epoch_continue_from = config['epoch_continue_from']
continue_trainings = int_to_boolean(config['continue_trainings'])
run_id = config['run_id']
save_model_and_optim_epochs = config['save_model_and_optim_epochs']
load_model_name = str(config['load_model_name'])
load_optim_name = str(config['load_optim_name'])
load_lr_name = str(config['load_lr_name'])

# Device configuration
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ' + str(device))

num_workers = config['num_workers']
pin_memory = config['pin_memory']

# Optimizer hyper-parameters 
optim_name = config['optim_name']
optim_default = int_to_boolean(config['optim_default'])
learning_rate = config['learning_rate']
rho = config['rho']
weight_decay = config['weight_decay']
momentum = config['momentum']
dampening = config['dampening']
nesterov = int_to_boolean(config['nesterov'])
scheduler_gamma = config['scheduler_gamma']

# Hyper-parameters 
num_epochs = config['num_epochs']
batch_size = config['batch_size']

# Data Augmentation
gaussian_blur = int_to_boolean(config['gaussian_blur'])
color_jitter = int_to_boolean(config['color_jitter'])
horizontal_flip = int_to_boolean(config['horizontal_flip'])
vertical_flip = int_to_boolean(config['vertical_flip'])
adjust_sharpness = int_to_boolean(config['adjust_sharpness'])
random_gamma = int_to_boolean(config['random_gamma'])
gaussian_noise = int_to_boolean(config['gaussian_noise'])
random_erasing = int_to_boolean(config['random_erasing'])
random_equalize = int_to_boolean(config['random_equalize'])
autocontrast = int_to_boolean(config['autocontrast'])
cut_paste_learn = int_to_boolean(config['cut_paste_learn'])

# Custom number of itearation per epoch
custom_len_epoch = config['custom_len_epoch']
activate_custom_epoch = int_to_boolean(config['activate_custom_epoch'])

# Model name
model_name = config['model_name']
trainable_backbone_layers = config['trainable_backbone_layers']
label_smoothing = config['label_smoothing']
gamma = config['gamma']
net_object = Model(model_name=model_name, trainable_backbone_layers=trainable_backbone_layers, label_smoothing = label_smoothing, gamma = gamma)
net = net_object.get_model()

# Load Model if continue training
if continue_trainings:
    print("Loading " + str(load_model_name))
    net = net_object.load_model(experiment_name, load_model_name)

# Data imbalance
data_imbalance_handler = int_to_boolean(config['data_imbalance_handler'])

print("Model layers:")
print(net)

net = net.to(device)
if seed != 0:
    torch.manual_seed(seed)

# Dataset
dataset_train = PVDefectsDS(get_transform(train=True, gaussian_blur = gaussian_blur, 
                            color_jitter=color_jitter, adjust_sharpness=adjust_sharpness, 
                            random_gamma=random_gamma, random_equalize=random_equalize, 
                            autocontrast=autocontrast, horizontal_flip=horizontal_flip, 
                            vertical_flip=vertical_flip, gaussian_noise=gaussian_noise, 
                            random_erasing=random_erasing, cut_paste_learn=cut_paste_learn), train_val_test = 0)
dataset_train_no_augmentation = PVDefectsDS(get_transform(train=False, gaussian_blur = False, 
                            color_jitter=False, adjust_sharpness=False, 
                            random_gamma=False, random_equalize=False, 
                            autocontrast=False, horizontal_flip=False, 
                            vertical_flip=False, gaussian_noise=False, 
                            random_erasing=False, cut_paste_learn=False), train_val_test = 0)
dataset_validation = PVDefectsDS(get_transform(train=False, gaussian_blur = False, 
                            color_jitter=False, adjust_sharpness=False, 
                            random_gamma=False, random_equalize=False, 
                            autocontrast=False, horizontal_flip=False, 
                            vertical_flip=False, gaussian_noise=False, 
                            random_erasing=False, cut_paste_learn=False), train_val_test = 1)
dataset_test = PVDefectsDS(get_transform(train=False, gaussian_blur = False, 
                            color_jitter=False, adjust_sharpness=False, 
                            random_gamma=False, random_equalize=False, 
                            autocontrast=False, horizontal_flip=False, 
                            vertical_flip=False, gaussian_noise=False, 
                            random_erasing=False, cut_paste_learn=False), train_val_test = 2)

# Handling data imbalance and Loading dataset
if data_imbalance_handler:
    sampler_object = Sampler(dataset = dataset_train)
    sampler = sampler_object.get_WeightedRandomSampler()
    trainloader = DataLoader(dataset_train, sampler = sampler, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn = collate_fn)
else:
    trainloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn = collate_fn)

trainloader_no_augmentation = DataLoader(dataset_train_no_augmentation, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn = collate_fn)
validationloader = DataLoader(dataset_validation, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn = collate_fn)
testloader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn = collate_fn)

# Loss function 
net_params = [p for p in net.parameters() if p.requires_grad]
optimizer_object = Optimizer(net_params,
                optim_name,
                optim_default,
                learning_rate,
                rho,
                weight_decay, # L2 penalty
                momentum,
                dampening,
                nesterov,
                scheduler_gamma)   
optimizer = optimizer_object.get_optim()
lr_scheduler = optimizer_object.get_lr_scheduler()

# Load Optim and lr_scheduler if continue training
if continue_trainings:
    print("Loading " + str(load_optim_name))
    optimizer = optimizer_object.load_optim(experiment_name, load_optim_name)
    print("Loading " + str(load_lr_name))
    lr_scheduler = optimizer_object.load_lr_scheduler(experiment_name, load_lr_name)

# Metrics
writer_training = Writer(experiment_name, run_name, 0, config, continue_trainings, run_id)
writer_validation = Writer(experiment_name, run_name, 1, config, continue_trainings, run_id)
writer_test = Writer(experiment_name, run_name, 2, config, continue_trainings, run_id)

# Create folders for saving paths
path = str("./states_saved/" + str(experiment_name))
if not continue_trainings:
    saved_models_path = str(path + "/saved_models")
    saved_optims_path = str(path + "/saved_opitm")
    saved_lr_path = str(path + "/saved_lr")
    os.makedirs(saved_models_path, exist_ok=True)
    os.makedirs(saved_optims_path, exist_ok=True)
    os.makedirs(saved_lr_path, exist_ok=True)

# Training
epoch = epoch_continue_from
iteration = 0
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

while epoch != num_epochs:
    # train for one epoch, printing every 10 iterations

    print("Starting training num." + str(epoch))
    # train for one epoch, printing every 10 iterations

    if activate_custom_epoch:
        iteration = custom_len_epoch*epoch
    else:
        iteration = len(trainloader)*epoch
    
    train_one_epoch(net, optimizer, trainloader, device, epoch, print_freq=100, iteration=iteration, writer = writer_training, activate_custom_epoch = activate_custom_epoch, custom_len_epoch = custom_len_epoch, model_name = model_name)
    loss_one_epoch_val(net, optimizer, validationloader, device, epoch, print_freq=100, iteration=iteration, writer = writer_validation, model_name = model_name)

    foldername_to_save_outputs = str("./runs/" + str(run_name) + "run_outputs_" + dt_string + "/epoch_" + str(epoch))
        
    # Log lr
    writer_training.store_metric(str('lr'), lr_scheduler.get_last_lr()[0], 'epoch', epoch)
    lr_scheduler.step()

    if epoch % evaluate_every_epochs == 0:
        print("Starting evaluation num. " + str(epoch))
        
        # evaluate on the train dataset
        print("Starting train data evaluation...")
        evaluate_engine(net,trainloader_no_augmentation, device, epoch, writer_training, foldername_to_save_outputs)
        
        # evaluate on the validation dataset
        print("Starting validation data evaluation...")
        evaluate_engine(net,validationloader, device, epoch, writer_validation, foldername_to_save_outputs)

        # evaluate on the test dataset
        print("Starting validation data evaluation...")
        evaluate_engine(net,testloader, device, epoch, writer_test, foldername_to_save_outputs)

        print("Computing confusion matrix training...")
        compute_confusion_matrix(epoch, writer_training, foldername_to_save_outputs, dataset_train_no_augmentation, iou_threshold = 0.5)

        print("Computing confusion matrix validation...")
        compute_confusion_matrix(epoch, writer_validation, foldername_to_save_outputs, dataset_validation, iou_threshold = 0.5)

        print("Removing cache memory...")
        shutil.rmtree(str(foldername_to_save_outputs + "_train"), ignore_errors=True)
        #shutil.rmtree(str(foldername_to_save_outputs + "_val"), ignore_errors=True)

    if epoch % save_model_and_optim_epochs == 0:
        print("Saving the model at epoch num. " + str(epoch))
        net_object.save_model(experiment_name, run_name, net, epoch)

        print("and saving optimizer and lr scheduler...")
        optimizer_object.save_optim_and_scheduler(experiment_name, run_name, optimizer, lr_scheduler, epoch)

    epoch += 1
        
print("That's it!")