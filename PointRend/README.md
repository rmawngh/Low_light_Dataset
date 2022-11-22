# PointRend

This is the implementation of the paper **PointRend: Image Segmentation as Rendering**

The original code is from [PointRend detectron2 Github](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)

# Requirements

This code is implemented with Python 3.6 (Anaconda)

```
Python == 3.6
CUDA >= 11.1
torch == 1.11.0
opencv-python
scikit-image
detectron2
```

# Pretrained Model
[PointRend](https://drive.google.com/file/d/10rnEzMteUd8Y0FJXJ2-Cgs-8Np2lC_Xm/view?usp=share_link)

# Code Desciption
You must change the absolute path of each Python file.

#### Create list file
```
python txt_list.py
```
output : list file
using this text file in my_dataset_function of each python file.

#### Training
```
python train_net.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --num-gpus 4
```

#### Evaluation
```
python train_net.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --eval-only MODEL.WEIGHTS /model_path/pointrend_model_final.pth
```
model_path : your model path

#### Visualization
```
python visualization.py
```

# Folder Description

```
configs       : PointRend config files
results       : Low light dataset result examples
pointrend     : PointRend model codes
```

# Results example

<p align="center">
<img src="https://user-images.githubusercontent.com/46700730/203223046-53f07eb7-37b4-4f0f-bb14-0a5129aa4f3e.gif">
</p>
