import cv2
import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config.config import CfgNode as CN
from detectron2.data.datasets import register_coco_instances



# Function for each feature evaluation and prediction
def predict_and_save_images(class_names_to_keep, output_folder):

    outside_features = ['WINDOW','PC','BRICK','TIMBER','LIGHT-ROOF','RC2-SLAB','RC2-COLUMN']
    inside_features = ['RC-SLAB','RC-JOIST','RC-COLUMN','TIMBER-COLUMN','TIMBER-JOIST']

    if class_names_to_keep[0] in inside_features:
        register_coco_instances("my_dataset_train", {}, f"/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/annotations/modified/train{'_'.join(class_names_to_keep)}.json", "/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/train")
        register_coco_instances("my_dataset_val", {}, f"/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/annotations/modified/val{'_'.join(class_names_to_keep)}.json", "/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/val")

    elif class_names_to_keep[0] in outside_features:
        register_coco_instances("my_dataset_train", {}, f"/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/annotations/modified/train{'_'.join(class_names_to_keep)}.json", "/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/train")
        register_coco_instances("my_dataset_val", {}, f"/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/annotations/modified/val{'_'.join(class_names_to_keep)}.json", "/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/val")
  
    ###############################################################
    #cfg setting parameters-----Begin
    cfg = get_cfg()
    cfg.NUM_GPUS = 1
    # Dataset Configuration
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    # Input Configuration
    cfg.INPUT.MIN_SIZE_TRAIN = (1024, 960)
    cfg.INPUT.MAX_SIZE_TEST = 1700
    cfg.INPUT.MAX_SIZE_TRAIN = 1000

    # Model Configuration
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.MASK_ON = True  # Ensure mask prediction is enabled

    # Anchor Generator Configuration
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    # Normalization Configuration
    cfg.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    # Solver Configuration
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.SOLVER.STEPS = (12000, 18000)
    cfg.SOLVER.GAMMA = 0.2

    # Evaluation Configuration
    cfg.TEST.EVAL_PERIOD = 2000  # Add this line to set the evaluation period

    # ResNeXt Configuration
    cfg.MODEL.RESNEXT = CN()
    cfg.MODEL.RESNEXT.DEPTH = 152
    cfg.MODEL.RESNEXT.NUM_GROUPS = 64
    cfg.MODEL.RESNEXT.WIDTH_PER_GROUP = 32

    # Output Configuration
    #cfg.OUTPUT_DIR = f"/scratch/tz2518/Segmentation_MRCNN/{class_names_to_keep}"
    cfg.OUTPUT_DIR = f"/scratch/tz2518/Segmentation_MRCNN/{class_names_to_keep}"

    #cfg setting parameters-----End
    ######--- For this part, both train.py and test.py should have the same parameters.


    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
    cfg.MODEL.RETINANET = False
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get("my_dataset_train")

    image_folder = f"/scratch/tz2518/test_all_models/data/{class_names_to_keep}"

    image_paths = [os.path.join(image_folder, image) for image in os.listdir(image_folder)]

    os.makedirs(output_folder, exist_ok=True)

    # Additional code to handle the creation of the output folder
    class_output_folder = os.path.join(output_folder, class_names_to_keep)
    os.makedirs(class_output_folder, exist_ok=True)

    image_paths = [os.path.join(image_folder, image) for image in os.listdir(image_folder) if os.path.splitext(image)[1] in ['.jpg', '.png']]
    for image_path in image_paths:
        print(f"Image path is: {image_path}")

        im = cv2.imread(image_path)

        outputs = predictor(im)

        prediction = outputs["instances"].pred_masks.shape[0] > 0
        #prediction = outputs["instances"].has("pred_masks")
        print(prediction)

        # Create a visualizer instance
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE)
        
        # Draw the instance predictions on the image
        instances = outputs["instances"].to("cpu")
        vis_output = v.draw_instance_predictions(predictions=instances)
        vis_image = vis_output.get_image()[:, :, ::-1]

        # Save the visualized image to the output folder
        output_image_path = os.path.join(class_output_folder, os.path.splitext(os.path.basename(image_path))[0] + '.jpg')
        cv2.imwrite(output_image_path, vis_image)

    



features = ["WINDOW", "PC", "BRICK", "LIGHT-ROOF", "RC2-SLAB", "RC2-COLUMN",
            "RC-SLAB", "RC-JOIST", "RC-COLUMN", "TIMBER-COLUMN", "TIMBER-JOIST"]

output_folder = "/scratch/tz2518/test_all_models/testoutput-Mask-RCNN"

for feature in features:
    predict_and_save_images(feature, output_folder)
