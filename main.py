import argparse
import sys

import os
import gc
import cv2
import math
import copy
import time
import random

import h5py
from PIL import Image
from io import BytesIO

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# For Image Models
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#-----------------------------------------------------------------------------------#

CONFIG = {
    "seed": 42,
    "img_size": 256,
    "valid_batch_size": 32,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])

#-------------------------------------------------------------------------------#
ROOT_DIR = "/kaggle/input/isic-2024-challenge"
TEST_CSV = f'{ROOT_DIR}/test-metadata.csv'
TEST_HDF = f'{ROOT_DIR}/test-image.hdf5'
SAMPLE = f'{ROOT_DIR}/sample_submission.csv'

#Best fold trained weights for each model - These models were trained on other Kaggle Notebooks
WEIGHT_effnetv1b0 = "/kaggle/input/isic2024-pytorch-all-models-training/Full_train_efficientnet_b0_final_AUROC0.9666_Loss0.2627_epoch20.bin"
WEIGHT_effvit = "/kaggle/input/isic2024-pytorch-all-models-training/Full_train_efficientvit_b0_2_final_AUROC0.9130_Loss0.3968_epoch19.bin"
WEIGHT_ghostnet = "/kaggle/input/isic2024-pytorch-all-models-training/Full_train_ghostnet_100_final_AUROC0.9679_Loss0.2483_epoch20.bin"
WEIGHT_mixnetS = "/kaggle/input/isic2024-pytorch-all-models-training/Full_train_mixnet_s_final_AUROC0.9722_Loss0.2587_epoch18.bin"
WEIGHT_mobilenetv2 = "/kaggle/input/isic2024-pytorch-all-models-training/Full_train_mobilenetv2_050_final_AUROC0.9459_Loss0.3309_epoch20.bin"

#Import metadata and submission file
df = pd.read_csv(TEST_CSV)[["isic_id", "patient_id"]]
df['target'] = 0 # dummy
df_sub = pd.read_csv(SAMPLE)

#-------------------------------------------------------------------------------#
class ISICDataset(Dataset):
    def __init__(self, df, file_hdf, transforms=None):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array( Image.open(BytesIO(self.fp_hdf[isic_id][()])) )
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target,
        }        
        
data_transforms = {
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.CenterCrop(224, 224),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class ISICModel_gem(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=False, checkpoint_path=None):
        super(ISICModel_gem, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))
        return output

class ISICModel_AVGpool(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=False, checkpoint_path=None):
        super(ISICModel_AVGpool, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.linear = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        logits = self.linear(features)
        output = self.sigmoid(logits)
        return output

class ISICModel_linear(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=False, checkpoint_path=None):
        super(ISICModel_linear, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, global_pool='avg')
        self.sigmoid = nn.Sigmoid()
    def forward(self, images):
        return self.sigmoid(self.model(images))


# Not all models had the same architecture, so we need to define different classes

gem_models = [
    # GEM pooling models
    ("tf_efficientnet_b0.ns_jft_in1k", "efficientnet_b0"),
    ("mixnet_s.ft_in1k", "mixnet_s"),
]

no_classifier_models = [
    # Models without a classifier
    ("efficientvit_b0.r224_in1k", "efficientvit_b0"),
]

default_models = [
    # Default pooling models
    ("mobilenetv2_050.lamb_in1k", "mobilenetv2_050"),
    ("ghostnet_100.in1k", "ghostnet_100"),
]

# Extract the first element of each tuple in gem_models
gem_model_names = [model[1] for model in gem_models]

# Extract the first element of each tuple in no_classifier_models
no_classifier_model_names = [model[1] for model in no_classifier_models]

# Extract the first element of each tuple in default_models
default_model_names = [model[1] for model in default_models]

# Display available models
print(f"Available Models: {gem_model_names + no_classifier_model_names + default_model_names}")

# Create loader
print("Creating loader...")
test_dataset = ISICDataset(df, TEST_HDF, transforms=data_transforms["valid"])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['valid_batch_size'], 
                          num_workers=2, shuffle=False, pin_memory=True)

# Define inference function
def inference(df, model_name, model_class, config, test_loader, weight_path):
    
    # Initialize the model
    model = model_class(model_name, pretrained=False)
    model.load_state_dict(torch.load(weight_path))
    model.to(config['device'])

    preds = []
    with torch.no_grad():
        bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for step, data in bar:        
            images = data['image'].to(CONFIG["device"], dtype=torch.float)        
            batch_size = images.size(0)
            outputs = model(images)
            preds.append( outputs.detach().cpu().numpy() )

    preds = np.concatenate(preds).flatten()

    # Create a unique column name based on the weight file
    weight_name = weight_path.split('/')[-1].replace('.bin', '')

    # Add predictions to the dataframe
    df[f'target_{weight_name}'] = preds

#---------------------------------------------------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on specified models")
    parser.add_argument("--models", nargs="+", help="List of model names to run inference on")
    args = parser.parse_args()

    all_models_dict = {
        "efficientnet_b0": WEIGHT_effnetv1b0,
        "mixnet_s": WEIGHT_mixnetS,
        "efficientvit_b0": WEIGHT_effvit,
        "mobilenetv2_050": WEIGHT_mobilenetv2,
        "ghostnet_100": WEIGHT_ghostnet
    }

    if args.models:
        selected_models = [(model, all_models_dict[model]) for model in args.models if model in all_models_dict]
    else:
        selected_models = list(all_models_dict.items())

    for model_name, weight_path in selected_models:
        if model_name in gem_model_names:
            model_class = ISICModel_gem
        elif model_name in no_classifier_model_names:
            model_class = ISICModel_linear
        elif model_name in default_model_names:
            model_class = ISICModel_AVGpool
        else:
            print(f"Unknown model type for {model_name}. Skipping.")
            continue
        
        print(f"Running inference on {model_name}...")
        inference(df, model_name, model_class, CONFIG, test_loader, weight_path)
        print(f"Inference completed for {model_name}.")
        print("---------------------------------------")
        
    # Save the predictions to a CSV file
    df.to_csv('test_preds_ER.csv', index=False)

# USAGE: python main.py --models efficientnet_b0 mixnet_s mobilenetv2_050 ghostnet_100