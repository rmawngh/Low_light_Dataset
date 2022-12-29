# Low-light Dataset Project

There are the two models used in Low-light Dataset Project.

Each folder describe their code, pretrained model and results.

<br>

The final low-light result dataset consists of 2 million images.

Train : Valid : Test = 8 : 1 : 1

Total : 2,043,025 (train : validation : test = 8 : 1 : 1)

Train : 1,634,420

Validation : 204,302

Test : 204,303

<br>

### TridentNet
Object Detection part used **Scale-Aware Trident Networks for Object Detection** paper.

The original code is from [TridentNet detectron2 Github](https://github.com/facebookresearch/detectron2/tree/main/projects/TridentNet)

### PointRend
Instance Segmentation part used **PointRend: Image Segmentation as Rendering** paper.

The original code is from [PointRend detectron2 Github](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)

<br>

### Requirements

This code is implemented with Python 3.6 (Anaconda)

```
Python == 3.6
CUDA >= 11.1
torch == 1.11.0
opencv-python
scikit-image
detectron2
```
