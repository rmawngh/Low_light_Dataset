#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config


import os, json
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
def read_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    data_list = []
    for i in range(len(lines)):
        data_list.append([lines[i].split(" ")[0],lines[i].split(" ")[1][:-1]])
    return data_list

def my_dataset_function(txt_dir="./half_data_train.txt"):
    data_files = read_text_lines(txt_dir)
    data_lens = len(data_files)
    dataset_dicts = []
    class_id = ['Book', 'Car', 'Scooter', 'Truck', 'Bus', 'Bicycle', 'Chair', 'Table', 'Dish', 'Flowerpot', 'Monitor',
                'Keyboard', 'Mouse', 'Weight', 'Motorcycle', 'Cup', 'Umbrella', 'Human', 'Boat', 'Frame', 'Bottle',
                'Laptop', 'Mirror', 'Ladle', 'Trash_Can', 'Pot', 'Cat', 'Clock', 'Kettle', 'Dog', 'Station', 'Ball',
                'Baseball_Glove', 'Camera', 'Calendar', 'Baseball_Bat', 'Racket', 'Bench', 'Stand_lamp', 'Handbag',
                'Glasses', 'Remote', 'Wallet', 'Smart_Phone', 'Suitcase', 'Can', 'Folding_Fan', 'Cap', 'Labacon',
                'Swing', 'Seesaw']
    for i in range(data_lens):
        img_file = "/ssd/JH" + data_files[i][0].split("media")[1] + "/Source_Data/" + data_files[i][1]
        bbox_json_file = "/ssd/JH" + data_files[i][0].split("media")[1] + "/Learning_Data/bbox/" \
                         + data_files[i][1][:-4] + ".json"
        poly_json_file = "/ssd/JH" + data_files[i][0].split("media")[1] + "/Learning_Data/poly/" \
                         + data_files[i][1][:-4] + ".json"
        with open(bbox_json_file) as f:
            bbox_imgs_anns = json.load(f)
        with open(poly_json_file) as f:
            poly_imgs_anns = json.load(f)
        record = {}
        record["file_name"] = os.path.join(img_file)
        record["height"] = int(poly_imgs_anns["Raw_Data_Info."]["Resolution"].split(", ")[1])
        record["width"] = int(poly_imgs_anns["Raw_Data_Info."]["Resolution"].split(", ")[0])
        record["image_id"] = i
        annos = poly_imgs_anns["Learning_Data_Info."]["Annotations"]
        b_annos = bbox_imgs_anns["Learning_Data_Info."]["Annotations"]
        objs = []
        for i, anno in enumerate(annos):
            bbox_p = b_annos[i]["Type_value"]
            class_id_temp = anno["Class_ID"]
            poly_p = anno["Type_value"]
            poly_num = int(len(poly_p)/2)
            poly_x = []
            poly_y = []
            for i in range(poly_num):
                poly_x.append(poly_p[i*2 + 0])
                poly_y.append(poly_p[i*2 + 1])
            poly = [(x + 0.5, y + 0.5) for x, y in zip(poly_x, poly_y)]
            poly = [p for x in poly for p in x]
            for i in range(len(class_id)):
                if class_id_temp == class_id[i]:
                    category_id = i
            obj = {
                "bbox": [bbox_p[0], bbox_p[1], bbox_p[2], bbox_p[3]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": category_id,
                "segmentation": [poly]
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts



def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.COLOR_AUG_SSD:
        augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        #evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        evaluator_type = 'coco'
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_instance":
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            return CityscapesSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    DatasetCatalog.register("my_train_dataset", lambda: my_dataset_function(txt_dir="./half_data_train.txt"))
    MetadataCatalog.get("my_train_dataset").thing_classes = ['Book', 'Car', 'Scooter', 'Truck', 'Bus', 'Bicycle', 'Chair', 'Table', 'Dish', 'Flowerpot', 'Monitor', 'Keyboard', 'Mouse', 'Weight', 'Motorcycle', 'Cup', 'Umbrella', 'Human', 'Boat', 'Frame', 'Bottle', 'Laptop', 'Mirror', 'Ladle', 'Trash_Can', 'Pot', 'Cat', 'Clock', 'Kettle', 'Dog', 'Station', 'Ball', 'Baseball_Glove', 'Camera', 'Calendar', 'Baseball_Bat', 'Racket', 'Bench', 'Stand_lamp', 'Handbag', 'Glasses', 'Remote', 'Wallet', 'Smart_Phone', 'Suitcase', 'Can', 'Folding_Fan', 'Cap', 'Labacon', 'Swing', 'Seesaw']


    DatasetCatalog.register("my_test_dataset", lambda: my_dataset_function(txt_dir="./half_data_test.txt"))
    MetadataCatalog.get("my_test_dataset").thing_classes = ['Book', 'Car', 'Scooter', 'Truck', 'Bus', 'Bicycle', 'Chair', 'Table', 'Dish', 'Flowerpot', 'Monitor', 'Keyboard', 'Mouse', 'Weight', 'Motorcycle', 'Cup', 'Umbrella', 'Human', 'Boat', 'Frame', 'Bottle', 'Laptop', 'Mirror', 'Ladle', 'Trash_Can', 'Pot', 'Cat', 'Clock', 'Kettle', 'Dog', 'Station', 'Ball', 'Baseball_Glove', 'Camera', 'Calendar', 'Baseball_Bat', 'Racket', 'Bench', 'Stand_lamp', 'Handbag', 'Glasses', 'Remote', 'Wallet', 'Smart_Phone', 'Suitcase', 'Can', 'Folding_Fan', 'Cap', 'Labacon', 'Swing', 'Seesaw']

    cfg.DATASETS.TRAIN = ("my_train_dataset",)
    cfg.DATASETS.TEST = ("my_test_dataset",)

    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
