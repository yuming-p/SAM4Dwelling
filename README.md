# Segment Anything Model Adapted for Dwelling Extraction (SAM4Dwelling) From High-Resolution Aerial Imagery
# SAM4Dwelling
apply SAM-Adapter on high resolution aerial images and drone images, from data preparation to prediction (work in progress)

this repository is adapted from [SAM4Refugee](https://github.com/YunyaGaoTree/SAM-Adapter-For-Refugee-Dwelling-Extraction) and [SAM-Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch/tree/main). 


## Data Preparation 
take input aerial image (RGB tiff file) and ground truth (shapefile with polygons), synchronise size of the two, segment aerial image into patches and upscale to 1024x1024 if needed

`python data_preparation/prepare_train_test.py --img-path --gt-path --dataname`

## Training
train with SAM-Adapter architecture, optimiser and learning rate scheduler is configurable, evaluation after each epoch is enabled. train and prediction can be run both on MPS backend for MacOS and CUDA GPU

`python run_sam/train.py --data`

## Prediction
use saved model state and optimizer for prediction

`python run_sam/predict.py --version --use-epoch`

## Dataset
- [Planetlab SkySat Collect](https://developers.planet.com/docs/data/skysat/)
- drone images from [OpenAerialMap](https://openaerialmap.org)
- hand drawn polygons in relatively small area of interest (each location as training data is around 2km $^2$)

  
