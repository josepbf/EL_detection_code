# Source: https://pytorch.org/vision/stable/models.html#object-detection
import torchvision
import torch
from datetime import datetime
import torch.nn.functional as F
from EL_train_utils import fastrcnn_loss_custom, retinanet_custom_loss
import functools

class Model:
    def __init__(self, 
                model_name,
                trainable_backbone_layers, 
                num_classes = 4, 
                pretrained = True, 
                pretrained_backbone = True,
                label_smoothing = 0.05,
                gamma = 2):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.trainable_backbone_layers = trainable_backbone_layers
        self.pretrained_backbone = pretrained_backbone
        self.label_smoothing = label_smoothing
        self.gamma = gamma

        if self.model_name == 'FasterRCNN_ResNet-50-FPN':
            # Faster R-CNN model with a ResNet-50-FPN backbone from the 
            # Faster R-CNN: Towards Real-Time Object Detection with Region 
            # Proposal Networks paper.
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                num_classes=num_classes, 
                pretrained_backbone=pretrained_backbone, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5

            if self.label_smoothing > 0:
                print("Applying label smoothing...")
                self.model.roi_heads.fastrcnn_loss = fastrcnn_loss_custom

        elif self.model_name == 'FasterRCNN_ResNet-50-FPN_v2':    
            # Constructs an improved Faster R-CNN model with a ResNet-50-FPN 
            # backbone from Benchmarking Detection Transfer Learning with Vision 
            # Transformers paper.
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                num_classes=num_classes, 
                pretrained_backbone=pretrained_backbone, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5

        elif self.model_name == 'FasterRCNN_MobileNetV3-Large':
            # Constructs a high resolution Faster R-CNN model with a MobileNetV3-Large 
            # FPN backbone.
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=pretrained, 
                num_classes=num_classes, 
                pretrained_backbone=pretrained_backbone, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 6
        
        elif self.model_name == 'RetinaNet_ResNet-50-FPN':
            # Constructs a RetinaNet model with a ResNet-50-FPN backbone.
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(
                pretrained_backbone = pretrained_backbone, 
                num_classes=num_classes, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5

            if self.gamma != 2:
                print("Modifying gamma...")
                self.model.compute_loss = retinanet_custom_loss

        elif self.model_name == 'RetinaNet_ResNet-50-FPN_v2':
            # Constructs an improved RetinaNet model with a ResNet-50-FPN backbone.
            self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                weights_backbone = 'DEFAULT',
                num_classes=num_classes, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5

        elif self.model_name == 'FCOS':
            # Constructs a FCOS model with a ResNet-50-FPN backbone.
            self.model = torchvision.models.detection.fcos_resnet50_fpn(
                pretrained_backbone = pretrained_backbone, 
                num_classes=num_classes, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5
                
        elif self.model_name == 'SSD':
             # The SSD300 model is based on the SSD: Single Shot MultiBox Detector paper.
            self.model = torchvision.models.detection.ssd300_vgg16(
                pretrained_backbone = pretrained_backbone, 
                num_classes=num_classes, 
                trainable_backbone_layers=trainable_backbone_layers) # Valid values are between 0 and 5
        
    def get_model(self):
        return self.model

    def load_model(self, experiment_name, load_model_name):
        self.model = torch.load("./states_saved/" + experiment_name + "/saved_models/" + str(load_model_name) + ".pth")
        return self.model

    def save_model(self, experiment_name, run_name, net, epoch):
        name_to_save = str("./states_saved/" + experiment_name + "/saved_models/" + self.model_name + "_" + run_name + "_epoch_" + str(epoch) + ".pth")
        torch.save(net, name_to_save)