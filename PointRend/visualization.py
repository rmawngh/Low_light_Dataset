import warnings
warnings.filterwarnings('ignore')

import detectron2
import os
import numpy as np
import cv2
from glob import glob
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
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

def my_dataset_function(txt_dir="./final_test.txt"):
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


DatasetCatalog.register("my_test_dataset", lambda: my_dataset_function(txt_dir="/ssd/JH/final_test.txt"))
MetadataCatalog.get("my_test_dataset").thing_classes = ['Book', 'Car', 'Scooter', 'Truck', 'Bus', 'Bicycle', 'Chair', 'Table', 'Dish', 'Flowerpot', 'Monitor', 'Keyboard', 'Mouse', 'Weight', 'Motorcycle', 'Cup', 'Umbrella', 'Human', 'Boat', 'Frame', 'Bottle', 'Laptop', 'Mirror', 'Ladle', 'Trash_Can', 'Pot', 'Cat', 'Clock', 'Kettle', 'Dog', 'Station', 'Ball', 'Baseball_Glove', 'Camera', 'Calendar', 'Baseball_Bat', 'Racket', 'Bench', 'Stand_lamp', 'Handbag', 'Glasses', 'Remote', 'Wallet', 'Smart_Phone', 'Suitcase', 'Can', 'Folding_Fan', 'Cap', 'Labacon', 'Swing', 'Seesaw']


my_dataset_metadata = MetadataCatalog.get("my_test_dataset")

test_dict = my_dataset_function(txt_dir="./final_test.txt")
#print(test_dict)

# import PointRend project
from detectron2.projects.point_rend import add_pointrend_config

#print(my_dataset_metadata)

SAVE_PATH = './result_images/visualization'
def main():
    #paths = glob('/ssd2/dataset/data_project_10/220606/Source_Data/*')
    for idx in range(len(test_dict)):#tqdm(paths):
        im = cv2.imread(test_dict[idx]["file_name"])
        
        im_orig = cv2.resize(im,dsize=(775,435), interpolation=cv2.INTER_AREA)
        cv2.imwrite('./result_images/original/' + test_dict[idx]["file_name"].split("/")[-1] , im_orig) 
        cfg = get_cfg()
        add_pointrend_config(cfg)
        cfg.merge_from_file("./configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = "./output/model_PointRend.pth"

        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)
        #print(outputs)
	
        v = Visualizer(im[:, :, ::-1], my_dataset_metadata, instance_mode=ColorMode.IMAGE_BW)
        point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        #print(point_rend_result.shape)
        _class, _name = test_dict[idx]["file_name"].split('/')[-2:]
        save_dir = os.path.join(SAVE_PATH, _class)
        #if os.path.isdir(save_dir) == False:
        #    os.mkdir(save_dir)
        plt.figure(figsize=(10,10))
        plt.imshow(point_rend_result)
        plt.axis('off')
        plt.savefig(os.path.join(SAVE_PATH, _name), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
   main()
   #  print(coco_metadata.get('thing_classes'))
