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


# Code Desciption
You must change the absolute path of each Python file.

#### Dataset list txt file

```
Total : 2,043,025 (train : validation : test = 8 : 1 : 1)
Train : 1,634,420
Validation : 204,302
Test : 204,303
```

[Train List](https://koreaoffice-my.sharepoint.com/:t:/g/personal/rmawngh_korea_ac_kr/EcS7Avk-PT1Lp6hCkNdnMlEBkmKyDiqhdd5mfafZTL97kQ?e=BnRZA9), 
[Validation List](https://koreaoffice-my.sharepoint.com/:t:/g/personal/rmawngh_korea_ac_kr/EZWcJgxm5ERFjbDFaalu0uUBniZmsytCZqfz9ITLxq_MHw?e=MCS4PF), 
[Test List](https://koreaoffice-my.sharepoint.com/:t:/g/personal/rmawngh_korea_ac_kr/ESktQjLFadJLj9IusAJLFBwBkvKiVcx5RTZGatxcOOYjFg?e=09znk8)


#### Training
```
python train_net.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --num-gpus 4
```

training setting
```
Class : 51
['Book', 'Car', 'Scooter', 'Truck', 'Bus', 'Bicycle', 'Chair', 'Table', 'Dish', 'Flowerpot', 'Monitor', 'Keyboard', 'Mouse', 'Weight', 'Motorcycle', 'Cup', 'Umbrella', 'Human', 'Boat', 'Frame', 'Bottle', 'Laptop', 'Mirror', 'Ladle', 'Trash_Can', 'Pot', 'Cat', 'Clock', 'Kettle', 'Dog', 'Station', 'Ball', 'Baseball_Glove', 'Camera', 'Calendar', 'Baseball_Bat', 'Racket', 'Bench', 'Stand_lamp', 'Handbag', 'Glasses', 'Remote', 'Wallet', 'Smart_Phone', 'Suitcase', 'Can', 'Folding_Fan', 'Cap', 'Labacon', 'Swing', 'Seesaw']
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

<p align="center">
<img alt="Left" width="200" height "200" src="https://user-images.githubusercontent.com/46700730/209906053-73a146ec-3f0b-4948-9d55-a8cfbc2ad99a.png">
<img alt="Right" width="200" height "200" src="https://user-images.githubusercontent.com/46700730/209906050-79a71a99-155f-4a64-a0f2-d93b8c397a22.png">
</p>
