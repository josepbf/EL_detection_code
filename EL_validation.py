import json
import time
import datetime
import copy
import numpy as np
import torchvision
import torch
import six
import os

import scipy.io
import math

from os import listdir
from os.path import isfile, join

import PIL
from PIL import Image

import collections
from collections import defaultdict, deque

from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from EL_utils import MetricLogger, SmoothedValue, reduce_dict, all_gather

def loss_one_epoch_val(model, optimizer, data_loader, device, epoch, print_freq, iteration, writer, model_name):
    metric_logger = MetricLogger(delimiter="  ")
    #metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, epoch, header):

        iteration = iteration + 1
        #print("Iteration: " + str(iteration))
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        loss = losses_reduced.item()

        if model_name == 'FasterRCNN_ResNet-50-FPN':
            loss_classifier = loss_dict_reduced['loss_classifier'].item()
            loss_box_reg = loss_dict_reduced['loss_box_reg'].item()
            loss_objectness = loss_dict_reduced['loss_objectness'].item()
            loss_rpn_box_reg = loss_dict_reduced['loss_rpn_box_reg'].item()

            writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss'), loss, str('iteration_' + writer.get_folder_name()), iteration)
            writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss_classifier'), loss_classifier, str('iteration_' + writer.get_folder_name()), iteration)
            writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss_box_reg'), loss_box_reg, str('iteration_' + writer.get_folder_name()), iteration)
            writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss_objectness'), loss_objectness, str('iteration_' + writer.get_folder_name()), iteration)
            writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss_rpn_box_reg'), loss_rpn_box_reg, str('iteration_' + writer.get_folder_name()), iteration)

        if model_name == 'SSD' or model_name == 'RetinaNet_ResNet-50-FPN' or model_name == 'FCOS':
            loss_classifier = loss_dict_reduced['classification'].item()
            loss_box_reg = loss_dict_reduced['bbox_regression'].item()

            writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss'), loss, str('iteration_' + writer.get_folder_name()), iteration)
            writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss_classifier'), loss_classifier, str('iteration_' + writer.get_folder_name()), iteration)
            writer.store_metric(str('Loss_' + writer.get_folder_name() + '/loss_box_reg'), loss_box_reg, str('iteration_' + writer.get_folder_name()), iteration)

def compute_confusion_matrix(epoch, writer, foldername_to_save_outputs, dataset, iou_threshold = 0.5):
  # Read all the names of outputs

    if writer.get_train_or_val() == 1:
        outputsPath = str(foldername_to_save_outputs + '_val')
    elif writer.get_train_or_val() == 0:
        outputsPath = str(foldername_to_save_outputs + '_train')
    
    #jsonPath = './FullTrainingDataset/Annotations'

    masks_path = "/zhome/de/6/201864/Downloads/PVDefectsDS/MasksVersionH-CorrAugust2023"

    hash_table = dataset.get_hash_names()
    # Remove leading zeros from the 'hash' column in the DataFrame
    hash_table['hash'] = hash_table['hash'].str.lstrip("0")
    hash_table['hash'] = hash_table['hash'].astype(str)
    outputNames = [f for f in listdir(outputsPath) if isfile(join(outputsPath, f))]
    outputNames.sort()

    score_thresholds = [0.5,0.7,0.9]
    for score_threshold in score_thresholds:
        confusion_matrix = torch.zeros(4,4)
        true_positive = 0
        positive = 0
        true = 0
        # totes les outputs
        for out_ind in range(len(outputNames)):

            # Read the output
            out = torch.load(str(outputsPath + '/' + outputNames[out_ind]))

            targetHash = outputNames[out_ind]
            targetHash = targetHash.replace(".pt","")
            result_df = hash_table.loc[hash_table['hash'] == targetHash]
            targetName = result_df.iloc[0]['image_name']


            imgs_names_dataset = dataset.get_imgs_names()
            row_index = imgs_names_dataset[imgs_names_dataset["namesAllCells"] == targetName].index[0]
            number_of_labels = int(imgs_names_dataset["nbDefAllCellsVH"].values[row_index])
            if number_of_labels != 0:
                mask_data = scipy.io.loadmat(masks_path + "/GT_" + str(targetName) + ".mat")
                labels_dataset = dataset.get_labels()
                row = labels_dataset.loc[labels_dataset["namesCellsWF"] == targetName]
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
            else:
                img_class = torch.tensor(0, dtype=torch.uint8)

            if img_class != 0:
                number_of_boxes = len(mask_data['GTLabelVH'])
                masks = mask_data['GTMaskVH']
                bbox = []
                labels = []

                # load images
                dataset_path = "/zhome/de/6/201864/Downloads/PVDefectsDS/"
                img_path = dataset_path + "/CellsImages/CellsGS/" + str(targetName[:5]) + "_" + str(targetName[5:12]) + "GS" + str(targetName[12:]) + ".png"
                img = Image.open(img_path)
                original_size = img.size[::-1]

                mask = masks
                if number_of_boxes > 1:
                    for i in range(number_of_boxes):
                        mask = masks[:,:,i]
                        xmin = math.trunc(min(np.where(mask != 0)[1]) * (300 / original_size[0])) 
                        xmax = math.trunc(max(np.where(mask != 0)[1]) * (300 / original_size[0]))
                        ymin = math.trunc(min(np.where(mask != 0)[0]) * (300 / original_size[1]))
                        ymax = math.trunc(max(np.where(mask != 0)[0]) * (300 / original_size[1]))

                        bbox.append((xmin, ymin, xmax, ymax))
                        labels.append(img_class)
                else:
                    xmin = math.trunc(min(np.where(mask != 0)[1]) * (300 / original_size[0]))
                    xmax = math.trunc(max(np.where(mask != 0)[1]) * (300 / original_size[0]))
                    ymin = math.trunc(min(np.where(mask != 0)[0]) * (300 / original_size[1]))
                    ymax = math.trunc(max(np.where(mask != 0)[0]) * (300 / original_size[1]))

                    bbox.append((xmin, ymin, xmax, ymax))
                    labels.append(img_class)
            else:
                bbox = []
                labels = []

            ground_truth_boxes = torch.tensor(bbox, dtype=torch.float)
            ground_truth_labels = torch.tensor(labels, dtype=torch.int64)

            # Compute the confusion matrix for that image

            # 1 Discard detections with IoU greater or equal than 0.5 Keep the highest score
            detected_boxes = out['boxes']
            detected_scores = out['scores']
            detected_labels = out['labels']
            indices = torchvision.ops.batched_nms(boxes = detected_boxes, scores = detected_scores, idxs = detected_labels, iou_threshold = iou_threshold)

            boxes = []
            scores = []
            labels = []
            for indice in indices:
                boxes.append(detected_boxes[indice])
                scores.append(detected_scores[indice])
                labels.append(detected_labels[indice])

            detected_boxes_prefilter = boxes
            detected_scores_prefilter = scores
            detected_labels_prefilter = labels

            # 2 Only detections with a score greater or equal than 0.5 are considered. Anything that’s under this value is discarded.
            # SCORE THRESHOLD IS A PARAM
            boxes = []
            scores = []
            labels = []
            for i in range(len(indices)):
                if detected_scores_prefilter[i] > score_threshold:
                    boxes.append(detected_boxes_prefilter[i])
                    scores.append(detected_scores_prefilter[i])
                    labels.append(detected_labels_prefilter[i])

            detected_boxes = boxes
            detected_scores = scores
            detected_labels = labels

            # 3 For each ground-truth box, the algorithm generates the IoU (Intersection over Union) with every detected box. 
            # A match is found if both boxes have an IoU greater or equal than 0.5.
            # IoU IS A PARAM
            boxes = []
            for detected_box in detected_boxes:
                boxes.append(torch.trunc(detected_box))

            detected_boxes = boxes

            boxes = []
            for detected_box in detected_boxes:
                xmin = detected_box[0].item()
                ymin = detected_box[1].item()
                xmax = detected_box[2].item()
                ymax = detected_box[3].item()
                boxes.append((xmin, ymin, xmax, ymax))

            detected_boxes = torch.tensor(boxes, dtype=torch.float)

            # no detections
            if len(detected_boxes) == 0:
                for m in range(len(ground_truth_labels)):
                    confusion_matrix[ground_truth_labels[m],0] += 1

            # no ground truth
            if len(ground_truth_boxes) == 0:
                for m in range(len(detected_labels)):
                    confusion_matrix[0,detected_labels[m]] += 1

            # detection and ground truth
            if len(ground_truth_boxes) != 0 and len(detected_boxes) != 0:
                iou = torchvision.ops.box_iou(boxes1 = ground_truth_boxes, boxes2 = detected_boxes)

                # 4 The list of matches is pruned to remove duplicates (ground-truth boxes that match with more than one detection box 
                # or vice versa). If there are duplicates, the best match (greater IoU) is always selected.
                detection_results = np.zeros(len(detected_boxes))
                ground_truth_results = np.zeros(len(ground_truth_boxes))
                
                for n in range(len(iou)):
                    for m in range(len(iou[0])):
                        if iou[n,m] >= iou_threshold:
                            index_detection = torch.argmax(input = iou[n], dim=None, keepdim=False) # Aquesta es l'index de la detecció
                            index_ground_truth = n # Aquest es l'index del ground truth
                        
                            # Només comptar ground_truth_results quan la detecció en questió encara no te resultat assignat
                            if detection_results[index_detection] == 0:
                                ground_truth_results[index_ground_truth] += 1
                            detection_results[index_detection] += 1
                        
                            if detection_results[index_detection] == 1 and ground_truth_results[index_ground_truth] == 1:
                                confusion_matrix[ground_truth_labels[index_ground_truth],detected_labels[index_detection]] += 1

                # 5 The confusion matrix is updated to reflect the resulting matches between ground-truth and detections.

                # 6 Objects that are part of the ground-truth but weren’t detected are counted in the last column of the matrix
                # (in the row corresponding to the ground-truth class). Objects that were detected but aren’t part of the confusion 
                # matrix are counted in the last row of the matrix (in the column corresponding to the detected class).
                for index_detection in range(len(detection_results)):
                    if detection_results[index_detection] == 0:
                        confusion_matrix[0,detected_labels[index_detection]] += 1

                for index_ground_truth in range(len(ground_truth_results)):
                    if ground_truth_results[index_ground_truth] == 0:
                        confusion_matrix[ground_truth_labels[index_ground_truth],0] += 1
            
            if out_ind == len(outputNames)-1 :
                print("Confunsion Matrix completed:")
                print("Score = " + str(score_threshold))
                print(confusion_matrix)

                true_positive = confusion_matrix[1,1] + confusion_matrix[2,2] + confusion_matrix[3,3]
                positive = true_positive + confusion_matrix[1,2] +confusion_matrix[1,3] + confusion_matrix[2,1] + confusion_matrix[2,3] +confusion_matrix[3,1] + confusion_matrix[3,2] +confusion_matrix[0,1]+confusion_matrix[0,2]+confusion_matrix[0,3]
                true = true_positive + confusion_matrix[1,2] +confusion_matrix[1,3] + confusion_matrix[2,1] + confusion_matrix[2,3] +confusion_matrix[3,1] + confusion_matrix[3,2] + confusion_matrix[1,0]+confusion_matrix[2,0]+confusion_matrix[3,0]
            
                if positive>0 and true>0:
                    if score_threshold == 0.5:
                        score_name = '50'
                        precision_50 = true_positive / positive
                        recall_50 = true_positive / true
                        fmeasure_50 = 2*(precision_50*recall_50)/(precision_50+recall_50)
                        print("Precision (score = 50): " + str(precision_50))
                        print("Recall (score = 50): " + str(recall_50))
                        print("Fmeasure (score = 50): " + str(fmeasure_50))
                        writer.store_metric('ConfusionMatrix' + writer.get_folder_name() + '/precision_0.5_score', precision_50, str('epoch_' + writer.get_folder_name()), epoch)
                        writer.store_metric('ConfusionMatrix' + writer.get_folder_name() + '/recall_0.5_score', recall_50, str('epoch_' + writer.get_folder_name()), epoch)
                        writer.store_metric('ConfusionMatrix' + writer.get_folder_name() + '/fmeasure_0.5_score', fmeasure_50, str('epoch_' + writer.get_folder_name()), epoch)
                    if score_threshold == 0.7:
                        score_name = '70'
                        precision_70 = true_positive / positive
                        recall_70 = true_positive / true
                        fmeasure_70 = 2*(precision_70*recall_70)/(precision_70+recall_70)
                        print("Precision (score = 70): " + str(precision_70))
                        print("Recall (score = 70): " + str(recall_70))
                        print("Fmeasure (score = 70): " + str(fmeasure_70))
                        writer.store_metric('ConfusionMatrix' + writer.get_folder_name() + '/precision_0.7_score', precision_70, str('epoch_' + writer.get_folder_name()), epoch)
                        writer.store_metric('ConfusionMatrix' + writer.get_folder_name() + '/recall_0.7_score', recall_70, str('epoch_' + writer.get_folder_name()), epoch)
                        writer.store_metric('ConfusionMatrix' + writer.get_folder_name() + '/fmeasure_0.7_score', fmeasure_70, str('epoch_' + writer.get_folder_name()), epoch)
                    if score_threshold == 0.9:
                        score_name = '90'        
                        precision_90 = true_positive / positive
                        recall_90 = true_positive / true
                        fmeasure_90 = 2*(precision_90*recall_90)/(precision_90+recall_90)  
                        print("Precision (score = 90): " + str(precision_90))
                        print("Recall (score = 90): " + str(recall_90))
                        print("Fmeasure (score = 90): " + str(fmeasure_90))
                        writer.store_metric('ConfusionMatrix' + writer.get_folder_name() + '/precision_0.9_score', precision_90, str('epoch_' + writer.get_folder_name()), epoch)
                        writer.store_metric('ConfusionMatrix' + writer.get_folder_name() + '/recall_0.9_score', recall_90, str('epoch_' + writer.get_folder_name()), epoch)
                        writer.store_metric('ConfusionMatrix' + writer.get_folder_name() + '/fmeasure_0.9_score', fmeasure_90, str('epoch_' + writer.get_folder_name()), epoch)
                # Store confusion matrix image
                writer.store_matrix('ConfusionMatrix' + writer.get_folder_name() + '/matrix', confusion_matrix, str('epoch_' + writer.get_folder_name()), epoch)

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

@torch.no_grad()
def evaluate_engine(model, data_loader, device, epoch, writer, foldername_to_save_outputs):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types, epoch)

    # directory save
    if writer.get_train_or_val() == 1:
        path = str(foldername_to_save_outputs + '_val')
        os.makedirs(path, exist_ok=True)
        output_directory_name = str(path + '/')
    elif writer.get_train_or_val() == 0:
        path = str(foldername_to_save_outputs + '_train')
        os.makedirs(path, exist_ok=True)
        output_directory_name = str(path + '/')
    elif writer.get_train_or_val() == 2:
        path = str(foldername_to_save_outputs + '_test')
        os.makedirs(path, exist_ok=True)
        output_directory_name = str(path + '/')


    for image, targets in metric_logger.log_every(data_loader, 100, epoch, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)


        outputs_to_save = outputs
        targets_copy = targets
        for i in range(len(outputs_to_save)):
          outputs_to_save[i]['image_id'] = targets_copy[i]['image_id']
          output_name = str(output_directory_name + str(targets_copy[i]['image_id'].item()) + '.pt')
          torch.save(outputs_to_save[i], output_name)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize(epoch, writer)
    torch.set_num_threads(n_threads)
    return coco_evaluator


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs

def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

def evaluate_coco(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

def createIndex(self):
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in self.dataset:
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in self.dataset:
        for img in self.dataset['images']:
            imgs[img['id']] = img

    if 'categories' in self.dataset:
        for cat in self.dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
        for ann in self.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats

def loadRes(self, resFile):
    """
    Load result file and return a result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object
    """
    res = COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    # print('Loading and preparing results...')
    # tic = time.time()
    if isinstance(resFile, six.string_types):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        'Results do not correspond to current coco set'
    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        for id, ann in enumerate(anns):
            ann['id'] = id + 1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if 'segmentation' not in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'segmentation' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            ann['area'] = maskUtils.area(ann['segmentation'])
            if 'bbox' not in ann:
                ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'keypoints' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            s = ann['keypoints']
            x = s[0::3]
            y = s[1::3]
            x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann['area'] = (x2 - x1) * (y2 - y1)
            ann['id'] = id + 1
            ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset['annotations'] = anns
    createIndex(res)
    return res

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks = targets['masks']
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            if 'masks' in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, epoch):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate_coco(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self, epoch, writer):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            
            coco_eval.summarize()
      
            mAP_IoU_50_95, mAP_IoU_50, mAR_IoU_50_95 = coco_eval.summarize_mAP_mAR()
            
            writer.store_metric('Accuracy_' + writer.get_folder_name() + '/mAP_IoU_50_95', mAP_IoU_50_95, str('epoch_' + writer.get_folder_name()), epoch)
            writer.store_metric('Accuracy_' + writer.get_folder_name() + '/mAP_IoU_50', mAP_IoU_50, str('epoch_' + writer.get_folder_name()), epoch)
            #writer.store_metric('Accuracy_' + writer.get_folder_name() + '/mAR_IoU_50_95', mAR_IoU_50_95, str('epoch_' + writer.get_folder_name()), epoch)

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def summarize_mAP_mAR(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            return mean_s
        
        mAP_IoU_50_95 = _summarize(1)
        mAP_IoU_50 = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        mAR_IoU_50_95 = _summarize(0, maxDets=self.params.maxDets[2])
        
        return mAP_IoU_50_95, mAP_IoU_50, mAR_IoU_50_95

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None


