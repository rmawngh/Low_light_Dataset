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

<p align="center">
<img alt="Left" width="400" height "400" src="https://user-images.githubusercontent.com/46700730/209906055-ddbb985a-28f8-4976-89f8-5fe6c68287d7.png">
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
python train_net.py --config-file configs/tridentnet_fast_R_50_C4_1x.yaml --num-gpus 4
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




# Docker (Ubuntu)

[TridentNet Docker](https://koreaoffice-my.sharepoint.com/:u:/g/personal/rmawngh_korea_ac_kr/EZxZKPuNdg9Cg91y-EokUfkBO0ISpszKPVchb2CpzRjsqg?e=EtsonM)

**Environment Setting**
```
install docker
install nvidia-docker

docker load -i TridentNet.tar
```

**activate docker container**
```
NV_GPU 0,1,2,3 nvidia-docker run --ipc=host -v /{low_light_dataset_path}:/dataset -ti low-light:latest

cd low-light/detectron/project/TridentNet
```
You should change the GPU number considering your computer settings.

**Train**
```
python3 train_net.py --config-file configs/tridentnet_fast_R_50_C4_1x.yaml --num-gpus 4
```

**Test**
```
python3 train_net.py --config-file configs/tridentnet_fast_R_50_C4_1x.yaml  --num-gpus 4 --eval-only MODEL.WEIGHTS ./output/model_TridentNet.pth
```

**Visualization**
```
python3 visualization.py
```

