# TridentNet

This is the implementation of the paper **Scale-Aware Trident Networks for Object Detection**

The original code is from [TridentNet detectron2 Github](https://github.com/facebookresearch/detectron2/tree/main/projects/TridentNet)

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
[TridentNet](https://koreaoffice-my.sharepoint.com/:u:/g/personal/rmawngh_korea_ac_kr/Ec6b50O6SMpPuVZsTZLI6dQBvaF6xpKN6OQ3SjbNRye6tQ?e=8TBlbV)

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
python train_net.py --config-file configs/tridentnet_fast_R_50_C4_1x.yaml --num-gpus 4
```

#### Evaluation
```
python train_net.py --config-file configs/tridentnet_fast_R_50_C4_1x.yaml --eval-only MODEL.WEIGHTS /model_path/tridentnet_model_final.pth
```
model_path : your model path

#### Visualization
```
python visualization.py
```

# Folder Description

```
configs       : TridentNet config files
results       : Low light dataset result examples
tridentnet    : TridentNet model codes
```

# Results example

<p align="center">
<img src="https://user-images.githubusercontent.com/46700730/203223053-45dd67c2-289c-4b45-b0e7-4ba1fd7a4353.gif">
</p>
