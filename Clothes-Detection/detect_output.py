import torch
import os
import cv2
from clothesdetection.yolo.utils.utils import *
from clothesdetection.predictors.YOLOv3 import YOLOv3Predictor
# from predictors.DetectronModels import Predictor
import glob
from tqdm import tqdm
import sys

def detect(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Top & Bottom definition
    top_cloth = [
        "short sleeve top",
        "long sleeve top",
        "short sleeve outwear",
        "long sleeve outwear",
        "vest",
        "sling"]

    bottom_cloth = [
        "shorts",
        "trousers",
        "skirt"]

    # YOLO PARAMS
    yolo_df2_params = {"model_def": "clothesdetection/yolo/df2cfg/yolov3-df2.cfg",
                       "weights_path": "clothesdetection/yolo/weights/yolov3-df2_15000.weights",
                       "class_path": "clothesdetection/yolo/df2cfg/df2.names",
                       "conf_thres": 0.5,
                       "nms_thres": 0.4,
                        "img_size": 416,
                        "device": device}

    yolo_modanet_params = {"model_def": "clothesdetection/yolo/modanetcfg/yolov3-modanet.cfg",
                           "weights_path": "clothesdetection/yolo/weights/yolov3-modanet_last.weights",
                           "class_path": "clothesdetection/yolo/modanetcfg/modanet.names",
                           "conf_thres": 0.5,
                           "nms_thres": 0.4,
                           "img_size": 416,
                        "device": device}

    # DATASET
    dataset = 'df2'

    if dataset == 'df2':  # deepfashion2
        yolo_params = yolo_df2_params

    if dataset == 'modanet':
        yolo_params = yolo_modanet_params

    # Classes
    classes = load_classes(yolo_params["class_path"])

    # Colors
    cmap = plt.get_cmap("rainbow")
    colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
    # np.random.shuffle(colors)



    model = model

    if model == 'yolo':
        detectron = YOLOv3Predictor(params=yolo_params)
    else:
        detectron = Predictor(model=model, dataset=dataset, CATEGORIES=classes)

    return detectron,classes,colors