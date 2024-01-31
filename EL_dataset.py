import torch 
from torch import Tensor
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torchvision.transforms.v2 import GaussianBlur, RandomEqualize, RandomAutocontrast, ColorJitter, RandomAdjustSharpness, RandomErasing
import scipy.io

import PIL
from PIL import Image, ImageFilter

import numpy as np
import pandas

import math
import random

from random import randrange

import os

import cv2

from ast import literal_eval as make_tuple

from os import listdir
from os.path import isfile, join

from EL_utils import string_to_dict, create_binary_mask, find_indexes, displace_mask, displace_image

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class OurRandomAutocontrast(object):
  def __init__(self, prob):
    self.prob = prob

  def __call__(self, image, target):
    randomautocontrast = RandomAutocontrast(self.prob)
    image = randomautocontrast(image)
    return image, target

class OurRandomEqualize(object):
  def __init__(self, prob):
    self.prob = prob

  def __call__(self, image, target):
    randomequalize = RandomEqualize(self.prob)
    image = randomequalize(image)
    return image, target

class OurRandomErasing(object):
  def __init__(self, prob):
    self.prob = prob
  def __call__(self, image, target):
    erasing = RandomErasing(self.prob)
    image = erasing(image)
    return image, target

class OurGaussianNoise(object):
    def __init__(self, prob):
        self.minstd = 0.0
        self.maxstd = 0.1
        self.mean = 0
        self.prob = prob
    def __call__(self, image, target):
      if random.random() < self.prob:
        std = np.random.uniform(self.minstd, self.maxstd)
        image = image + torch.randn(image.size()) * std + self.mean
      return image, target

class OurRandomGamma(object):
    def __init__(self, prob):
        self.prob = prob
        mingamma = 4/5
        maxgamma = 5/4
        self.minloggamma = np.log(mingamma)
        self.maxloggamma = np.log(maxgamma)
    def __call__(self, image, target):
      if random.random() < self.prob:
        gamma = np.exp(np.random.uniform(self.minloggamma, self.maxloggamma))
        image = TF.adjust_gamma(image, gamma=gamma)
      return image, target

class OurColorJitter(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            jitter = ColorJitter(brightness=0.25, contrast=0.25)
            image = jitter(image)

        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob
    # 0: xmin, 1: ymin, 2: xmax, 3: ymax
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]

        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]

        return image, target

class CutPasteAndLearn(object):
    def __init__(self, prob):
        self.prob = prob 
        self.UCF_multi_path = "/zhome/de/6/201864/Downloads/UCF-EL-Defect-main/Multi"
        self.onlyfiles = [f for f in listdir(self.UCF_multi_path) if isfile(join(self.UCF_multi_path, f))]
        self.annotationsUCF = pandas.read_csv("/zhome/de/6/201864/Downloads/UCF-EL-Defect-main/AnnotationsCombined.csv")

    def __call__(self, image, target):
        class_original_image = target['labels'][0]
        if class_original_image == 0:
          if random.random() < self.prob:
            image_to_read_index = randrange(len(self.onlyfiles))
            imageName = self.onlyfiles[image_to_read_index]
            
            selected_rows = self.annotationsUCF[self.annotationsUCF['filename'] == imageName]
            selected_rows_2 = selected_rows[selected_rows['region_shape_attributes'].str.contains("polygon", case=False, na=False)]
            desired_strings = ["Crack_Resistive", "Crack_Isolated"]
            selected_rows_3 = selected_rows_2[selected_rows_2['region_attributes'].str.contains("Crack_Resistive", case=False, na=False) | selected_rows_2['region_attributes'].str.contains("Crack_Isolated", case=False, na=False)]
            selected_rows_3_list = selected_rows_3['region_shape_attributes'].tolist()
            selected_rows_3_list_1 = selected_rows_3['region_attributes'].tolist()
            
            while len(selected_rows_3_list) == 0:
              image_to_read_index = randrange(len(self.onlyfiles))
              imageName = self.onlyfiles[image_to_read_index]

              selected_rows = self.annotationsUCF[self.annotationsUCF['filename'] == imageName]
              selected_rows_2 = selected_rows[selected_rows['region_shape_attributes'].str.contains("polygon", case=False, na=False)]
              desired_strings = ["Crack_Resistive", "Crack_Isolated"]
              selected_rows_3 = selected_rows_2[selected_rows_2['region_attributes'].str.contains("Crack_Resistive", case=False, na=False) | selected_rows_2['region_attributes'].str.contains("Crack_Isolated", case=False, na=False)]
              selected_rows_3_list = selected_rows_3['region_shape_attributes'].tolist()
              selected_rows_3_list_1 = selected_rows_3['region_attributes'].tolist()

            image_to_read_index = randrange(len(selected_rows_3_list))

            polygon_dirty = selected_rows_3_list[image_to_read_index]
            label_dirty = selected_rows_3_list_1[image_to_read_index]

            dictionary_label = string_to_dict(label_dirty)
            dictionary_polygon = string_to_dict(polygon_dirty)

            if dictionary_label['Defect_Class'] == 'Crack_Resistive':
              img_class = 1 #Crack A/B
            elif dictionary_label['Defect_Class'] == 'Crack_Isolated':
              img_class = 2 #Crack C
            else:
              print("Something is wrong on the UCF dataset...")

            image_UCF = Image.open(self.UCF_multi_path + "/" + imageName)
            image_UCF = torchvision.transforms.functional.rotate(image_UCF, 90)


            binary_mask = create_binary_mask(dictionary_polygon['all_points_x'], dictionary_polygon['all_points_y'], (image.size[1], image.size[0]))
            binary_mask = torchvision.transforms.functional.rotate(binary_mask, 90)
            binary_mask = np.array(binary_mask)
            binary_mask = cv2.resize(binary_mask, (300, 300))
            image_resized = image_UCF.resize((300, 300))

            min_index, max_index = find_indexes(binary_mask, 1)

            dx_max = -(300-max_index[0])
            dx_min = min_index[0]
            dy_max = -(300-max_index[1])
            dy_min = min_index[1]
              
            dx = randrange(dx_max, dx_min)
            dy = randrange(dy_max, dy_min)

            binary_mask_displaced = displace_mask(binary_mask, dy, dx)

            # Convert the image to a NumPy array
            image_array = np.array(image_resized)
            image_displaced = displace_image(image_array, dy, dx)

            # Create a PIL Image from the matrix
            image_displaced_pil = Image.fromarray(image_displaced.astype('uint8')).convert("L")

            # Create a PIL image from the binary matrix
            binary_mask_displaced_pil = Image.fromarray(binary_mask_displaced * 255, mode='L')

            binary_mask_displaced_pil_blur = binary_mask_displaced_pil.filter(ImageFilter.GaussianBlur(2))

            image_composed = Image.composite(image_displaced_pil, image, binary_mask_displaced_pil_blur)

            min_index, max_index = find_indexes(binary_mask_displaced, 1)

            xmax = max_index[1]
            xmin = min_index[1]
            ymax = max_index[0]
            ymin = min_index[0]
            bbox = []
            labels = []
            areas = []
            area = (xmax-xmin)*(ymax-ymin)
            bbox.append((xmin, ymin, xmax, ymax))
            areas.append(area)
            labels.append(img_class)

            number_of_boxes = 1
            bbox = torch.tensor(bbox, dtype=torch.float)
            areas = torch.tensor(areas, dtype=torch.int64)
            labels = torch.tensor(labels, dtype=torch.int64)
            iscrowd = torch.zeros((number_of_boxes,), dtype=torch.int64) # suppose all instances are not crowd

            original_target = target

            target = {}
            target['boxes'] = bbox
            target['labels'] = labels
            target['area'] = areas
            target['image_id'] = original_target['image_id']
            target["iscrowd"] = iscrowd

            image = image_composed

        return image, target

class OurGaussianBlur(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            blurfilter = GaussianBlur(kernel_size=5, sigma=(0.01,1.0))
            image = blurfilter(image)
        return image, target

class OurRandomAdjustSharpness(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image, target):
        randomadjustsharpnessfilter = RandomAdjustSharpness(sharpness_factor=1.5, p=self.prob)
        image = randomadjustsharpnessfilter(image)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def get_transform(train, gaussian_blur, color_jitter, adjust_sharpness, random_gamma, random_equalize, 
                      autocontrast, horizontal_flip, vertical_flip, gaussian_noise, random_erasing, cut_paste_learn):
  available_transforms = []
  transforms = []
  if train:
    if cut_paste_learn:
      transforms.append(CutPasteAndLearn(0.125))
    if gaussian_blur:
      available_transforms.append(OurGaussianBlur(0.5))
    if color_jitter:
      available_transforms.append(OurColorJitter(0.5))
    if adjust_sharpness:
      available_transforms.append(OurRandomAdjustSharpness(0.5))
    if random_gamma:
      available_transforms.append(OurRandomGamma(0.5))
    if random_equalize:
      available_transforms.append(OurRandomEqualize(0.5))
    if autocontrast:
      available_transforms.append(OurRandomAutocontrast(0.5))
  
  if available_transforms:
    selected_transform = random.choice(available_transforms)
    transforms.append(selected_transform)
  transforms.append(ToTensor())

  if train:
    if horizontal_flip:
      transforms.append(RandomHorizontalFlip(0.5))
    if vertical_flip:
      transforms.append(RandomVerticalFlip(0.5))
    if gaussian_noise:
      transforms.append(OurGaussianNoise(0.5))
    if random_erasing:
      transforms.append(OurRandomErasing(0.5))
  return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

class PVDefectsDS(torch.utils.data.Dataset):
    def __init__(self, transforms, train_val_test):
        self.transforms = transforms
        self.labels = list()
        self.imgs_names = list()
        self.train_val_test = train_val_test
        
        self.dataset_path = "/zhome/de/6/201864/Downloads/PVDefectsDS/"
        annotationsPath = self.dataset_path + "namesDSVersionH_CorrAug2023.xlsx"
        
        self.masks_path = self.dataset_path + 'MasksVersionH-CorrAugust2023/'
        print("Loading dataset...")

        for i in range(12):
          if i%2 == 0:
            self.labels.append(pandas.read_excel(annotationsPath,i)[:-2])
          else:
            self.imgs_names.append(pandas.read_excel(annotationsPath,i)[:-2])
        
        self.imgs_names = pandas.concat(self.imgs_names, ignore_index=True)
        self.labels = pandas.concat(self.labels, ignore_index=True)

        split_train_val_test_csv = pandas.read_csv(self.dataset_path + 'split_train_val_test.csv')

        self.split_imgs_name = []
        if train_val_test == 0:
          # Train
          for i in range(len(split_train_val_test_csv)):
            if split_train_val_test_csv.at[i,'train_val_test_split'] == 0:
              self.split_imgs_name.append(split_train_val_test_csv.at[i,"name_image"])
        elif train_val_test == 1:
          # Validation
          for i in range(len(split_train_val_test_csv)):
            if split_train_val_test_csv.at[i,'train_val_test_split'] == 1:
              self.split_imgs_name.append(split_train_val_test_csv.at[i,"name_image"])
        elif train_val_test == 2:
          # Test
          for i in range(len(split_train_val_test_csv)):
            if split_train_val_test_csv.at[i,'train_val_test_split'] == 2:
              self.split_imgs_name.append(split_train_val_test_csv.at[i,"name_image"])
        
        # Create hash name table
        columns = ['image_name', 'hash']
        self.hash_names_table = pandas.DataFrame(columns=columns)
        for i in range(len(self.split_imgs_name)):
          hash_image_name = str(hash(self.split_imgs_name[i]))
          hash_image_name = hash_image_name[1:18]
          new_row = {'image_name': self.split_imgs_name[i], 'hash': hash_image_name}
          self.hash_names_table = pandas.concat([self.hash_names_table, pandas.DataFrame([new_row])], ignore_index=True)

    def __getitem__(self, idx):
        # define paths
        self.dataset_path = "/zhome/de/6/201864/Downloads/PVDefectsDS/"
        img_path = self.dataset_path + "/CellsImages/CellsGS/" + str(self.split_imgs_name[idx][:5]) + "_" + str(self.split_imgs_name[idx][5:12]) + "GS" + str(self.split_imgs_name[idx][12:]) + ".png"
        row_index = self.imgs_names[self.imgs_names["namesAllCells"] == self.split_imgs_name[idx]].index[0]
        number_of_labels = int(self.imgs_names["nbDefAllCellsVH"].values[row_index])
        if number_of_labels != 0:
          row = self.labels.loc[self.labels["namesCellsWF"] == self.split_imgs_name[idx]]
          if row['nbCAVH'].values[0] > 0:
            img_class = torch.tensor(1, dtype=torch.uint8)
          elif row['nbCBVH'].values[0] > 0:
            img_class = torch.tensor(1, dtype=torch.uint8)
          elif row['nbCCVH'].values[0] > 0:
            img_class = torch.tensor(2, dtype=torch.uint8)
          elif row['nbFFVH'].values[0] > 0:
            img_class = torch.tensor(3, dtype=torch.uint8)
          else:
            print("Image not labeled correctly")
            print(self.imgs_names["namesAllCells"])
        else:
          img_class = torch.tensor(0, dtype=torch.uint8)

        # load images
        img = Image.open(img_path)
        original_size = img.size[::-1]
        img = torchvision.transforms.functional.resize(img, (300,300))

        image_name = self.split_imgs_name[idx]
        image_id = str(hash(image_name))
        image_id = image_id[1:18]
        image_id = int(image_id)
        image_id = torch.tensor(image_id, dtype=torch.int64)

        if img_class != 0:

          mask_data = scipy.io.loadmat(self.masks_path + "GT_" + str(self.split_imgs_name[idx]) + ".mat")
          number_of_boxes = len(mask_data['GTLabelVH'])
          masks = mask_data['GTMaskVH']
          bbox = []
          labels = []
          areas = []

          mask = masks
          if number_of_boxes > 1:
            for i in range(number_of_boxes):
              mask = masks[:,:,i]
              xmin = math.trunc(min(np.where(mask != 0)[1]) * (300 / original_size[0])) 
              xmax = math.trunc(max(np.where(mask != 0)[1]) * (300 / original_size[0]))
              ymin = math.trunc(min(np.where(mask != 0)[0]) * (300 / original_size[1]))
              ymax = math.trunc(max(np.where(mask != 0)[0]) * (300 / original_size[1]))

              area = (xmax-xmin)*(ymax-ymin)
              bbox.append((xmin, ymin, xmax, ymax))
              areas.append(area)
              labels.append(img_class)
          else:
            xmin = math.trunc(min(np.where(mask != 0)[1]) * (300 / original_size[0]))
            xmax = math.trunc(max(np.where(mask != 0)[1]) * (300 / original_size[0]))
            ymin = math.trunc(min(np.where(mask != 0)[0]) * (300 / original_size[1]))
            ymax = math.trunc(max(np.where(mask != 0)[0]) * (300 / original_size[1]))

            area = (xmax-xmin)*(ymax-ymin)
            bbox.append((xmin, ymin, xmax, ymax))
            areas.append(area)
            labels.append(img_class)

          bbox = torch.tensor(bbox, dtype=torch.float)
          areas = torch.tensor(areas, dtype=torch.int64)
          labels = torch.tensor(labels, dtype=torch.int64)
          iscrowd = torch.zeros((number_of_boxes,), dtype=torch.int64) # suppose all instances are not crowd

          target = {}
          target['boxes'] = bbox
          target['labels'] = labels
          target['area'] = areas
          target['image_id'] = image_id
          target["iscrowd"] = iscrowd

        else:
          target = {}
          bbox = []
          bbox.append((0, 0, 300, 300))
          bbox = torch.tensor(bbox, dtype=torch.float)
          target['boxes'] = bbox
          labels = []
          labels.append(0)
          labels = torch.tensor(labels, dtype=torch.int64)
          target['labels'] = labels
          area = 300*300
          areas = []
          areas.append(area)
          areas = torch.tensor(areas, dtype=torch.int64)
          target['area'] = areas
          target['image_id'] = torch.tensor(image_id, dtype=torch.int64)
          iscrowd = torch.zeros((1,), dtype=torch.int64) # suppose all instances are not crowd
          target["iscrowd"] = iscrowd

        if self.transforms is not None:
          img, target = self.transforms(img,target)

        return img, target

    def __len__(self):
      return len(self.split_imgs_name)

    def get_hash_names(self):
      return self.hash_names_table
    
    def get_imgs_names(self):
      return self.imgs_names

    def get_labels(self):
      return self.labels

class Sampler:
  def __init__(self, dataset):
    self.dataset = dataset

  def get_WeightedRandomSampler(self):
    # Initialize an empty list to store class labels
    all_labels = []

    # Iterate through the dataset to collect all labels
    for idx in range(len(self.dataset)):
      _, target = self.dataset[idx]
      label = target['labels'][0].item()  # Extract the label from target
      all_labels.append(label)

    # Convert the list to a PyTorch tensor
    all_labels_tensor = torch.tensor(all_labels)

    # Calculate class counts
    class_counts = torch.bincount(all_labels_tensor)

    # Initialize an empty list to store weights for each sample
    weights_per_sample = []
    for idx in range(len(self.dataset)):
      _, target = self.dataset[idx]
      label = target['labels'][0].item()  # Extract the label from target
      class_weight = 1.0 / class_counts[label].float()  # Inverse class frequency
      weights_per_sample.append(class_weight)

    # Create the sampler 
    # Samples elements from [0,..,len(weights)-1] with given probabilities (weights). 
    return torch.utils.data.WeightedRandomSampler(weights=weights_per_sample, num_samples=len(self.dataset), replacement=True)
