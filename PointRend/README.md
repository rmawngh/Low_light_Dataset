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

[PointRend](https://koreaoffice-my.sharepoint.com/:u:/g/personal/rmawngh_korea_ac_kr/EflwSSg0WgFHhyC8NTYh56wBjq16hctngYRJV5-4hdSnuw?e=rX64he)


<p align="center">
<img alt="Left" width="400" height "400" src="https://user-images.githubusercontent.com/46700730/209906053-73a146ec-3f0b-4948-9d55-a8cfbc2ad99a.png">
<img alt="Right" width="400" height "400" src="https://user-images.githubusercontent.com/46700730/209906050-79a71a99-155f-4a64-a0f2-d93b8c397a22.png">
</p>

# Code Desciption
You must change the absolute path of each Python file.

#### Dataset list txt file

```
Total : 2,036,572 (train : validation : test = 8 : 1 : 1)
Train : 1,629,260
Validation : 203,656
Test : 203,656
```

[Train List](https://koreaoffice-my.sharepoint.com/:t:/g/personal/rmawngh_korea_ac_kr/EVZgGXG5oDNDuJuyKo03ywMBC7ukbs4H1AJWMX_OyTbsmQ?e=FXjqGX), 
[Validation List](https://koreaoffice-my.sharepoint.com/:t:/g/personal/rmawngh_korea_ac_kr/EUXdFbd6nHJGmR97xjzGFBwBKnAIGnlRltCDs0EqPC9yzA?e=JVxltG), 
[Test List](https://koreaoffice-my.sharepoint.com/:t:/g/personal/rmawngh_korea_ac_kr/EaMC1M4Z2bRHm9I7LQ-Rp7EBa4AyaJfJ_4-RE_4cLQTJOQ?e=L2dOFO)


#### Training
```
python train_net.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --num-gpus 4
```

training setting
```
Class : 51
Batch size : 16
iteration : 250,000
iteration step : (180,000, 220,000)
```


#### Evaluation
```
python train_net.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --eval-only MODEL.WEIGHTS /model_path/model_PointRend.pth
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



# Docker (Ubuntu)

[PointRend Docker](https://koreaoffice-my.sharepoint.com/:u:/g/personal/rmawngh_korea_ac_kr/EUfzXuMOVD9BgZinLwSbxMQBVT7kh41lYLoTrIachpH-ow?e=uhfKyY)

**Environment Setting**
```
install docker
install nvidia-docker

docker load -i PointRend.tar
```

**activate docker container**
```
NV_GPU 0,1,2,3 nvidia-docker run --ipc=host -v /{low_light_dataset_path}:/dataset -ti low-light:latest

cd low-light/detectron/project/PointRend
```
You should change the GPU number considering your computer settings.

**Train**
```
python3 train_net.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --num-gpus 4
```

**Test**
```
python3 train_net.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml  --num-gpus 4 --eval-only MODEL.WEIGHTS ./output/model_PointRend.pth
```

**Visualization**
```
python3 visualization.py
```
