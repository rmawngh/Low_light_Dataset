#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator

from tridentnet import add_tridentnet_config


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

def my_dataset_function(txt_dir="./final_train.txt"):
    #json_file = os.path.join(img_dir, "via_region_data.json")

    data_files = read_text_lines(txt_dir)

    data_lens = len(data_files)

    dataset_dicts = []

    class_id = ['Book', 'Car', 'Scooter', 'Truck', 'Bus', 'Bicycle', 'Chair', 'Table', 'Dish', 'Flowerpot', 'Monitor', 'Keyboard', 'Mouse', 'Weight', 'Motorcycle', 'Cup', 'Umbrella', 'Human', 'Boat', 'Frame', 'Bottle', 'Laptop', 'Mirror', 'Ladle', 'Trash_Can', 'Pot', 'Cat', 'Clock', 'Kettle', 'Dog', 'Station', 'Ball', 'Baseball_Glove', 'Camera', 'Calendar', 'Baseball_Bat', 'Racket', 'Bench', 'Stand_lamp', 'Handbag', 'Glasses', 'Remote', 'Wallet', 'Smart_Phone', 'Suitcase', 'Can', 'Folding_Fan', 'Cap', 'Labacon', 'Swing', 'Seesaw']

    for i in range(data_lens):
        first = data_files[i][0].split("Source_data")[0]
        second = data_files[i][0].split("Source_data")[1]
        img_file = first + "/Source_data/" + second[:-4] + ".jpg"
        bbox_json_file = first + "/Bounding_Box/" + second[:-4] + ".json"
        poly_json_file = first + "/Segmentation/" + second[:-4] + ".json"

        with open(bbox_json_file) as f:
            bbox_imgs_anns = json.load(f)
        with open(poly_json_file) as f:
            poly_imgs_anns = json.load(f)

        record = {}
        record["file_name"] = os.path.join(img_file)
        record["height"] = int(bbox_imgs_anns["Raw_Data_Info."]["Resolution"].split(", ")[1])
        record["width"] = int(bbox_imgs_anns["Raw_Data_Info."]["Resolution"].split(", ")[0])
        record["image_id"] = i

        annos = bbox_imgs_anns["Learning_Data_Info."]["Annotations"]
        objs = []
        for anno in annos:
            bbox_p = anno["Type_value"]
            class_id_temp = anno["Class_ID"]
            for i in range(len(class_id)):
                if class_id_temp == class_id[i]:
                    category_id = i

            obj = {
                "bbox": [bbox_p[0], bbox_p[1], bbox_p[2], bbox_p[3]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": category_id,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    DatasetCatalog.register("my_train_dataset", lambda: my_dataset_function(txt_dir="./final_train.txt"))
    MetadataCatalog.get("my_train_dataset").thing_classes = ['Book', 'Car', 'Scooter', 'Truck', 'Bus', 'Bicycle', 'Chair', 'Table', 'Dish', 'Flowerpot', 'Monitor', 'Keyboard', 'Mouse', 'Weight', 'Motorcycle', 'Cup', 'Umbrella', 'Human', 'Boat', 'Frame', 'Bottle', 'Laptop', 'Mirror', 'Ladle', 'Trash_Can', 'Pot', 'Cat', 'Clock', 'Kettle', 'Dog', 'Station', 'Ball', 'Baseball_Glove', 'Camera', 'Calendar', 'Baseball_Bat', 'Racket', 'Bench', 'Stand_lamp', 'Handbag', 'Glasses', 'Remote', 'Wallet', 'Smart_Phone', 'Suitcase', 'Can', 'Folding_Fan', 'Cap', 'Labacon', 'Swing', 'Seesaw']

    my_dataset_metadata = MetadataCatalog.get("my_train_dataset")
    
    DatasetCatalog.register("my_test_dataset", lambda: my_dataset_function(txt_dir="./final_test.txt"))
    MetadataCatalog.get("my_test_dataset").thing_classes = ['Book', 'Car', 'Scooter', 'Truck', 'Bus', 'Bicycle', 'Chair', 'Table', 'Dish', 'Flowerpot', 'Monitor', 'Keyboard', 'Mouse', 'Weight', 'Motorcycle', 'Cup', 'Umbrella', 'Human', 'Boat', 'Frame', 'Bottle', 'Laptop', 'Mirror', 'Ladle', 'Trash_Can', 'Pot', 'Cat', 'Clock', 'Kettle', 'Dog', 'Station', 'Ball', 'Baseball_Glove', 'Camera', 'Calendar', 'Baseball_Bat', 'Racket', 'Bench', 'Stand_lamp', 'Handbag', 'Glasses', 'Remote', 'Wallet', 'Smart_Phone', 'Suitcase', 'Can', 'Folding_Fan', 'Cap', 'Labacon', 'Swing', 'Seesaw']

    cfg.DATASETS.TRAIN = ("my_train_dataset",)
    cfg.DATASETS.TEST = ("my_test_dataset",)
    #print(args)
    add_tridentnet_config(cfg)
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
