import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import json
from torch.autograd import Variable
import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Name of the trained model')
parser.add_argument('--top_k', type=int, default=1, help='Number of classes you wish to see in descending order')
parser.add_argument('--json', type=str, default='cat_to_name.json', help='Json file holding class names')
parser.add_argument('--gpu', type=bool, help='True = gpu, False = cpu')
parser.add_argument('--image_path', type=str, default='flowers/test/100/image_07899.jpg', help='Location of image to predict')
args = parser.parse_args()

if args.gpu == True:
    print('This program is running on gpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    print('This program is running on cpu')
    device = torch.device('cpu')
    
filepath = args.json
with open(filepath, 'r') as f:
    cat_to_name = json.load(f)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    
    img = Image.open(image)
    img = img.resize((256, 256))
    img = img.crop((0, 0, 224, 224))
    img = np.array(img)/255
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - means) / std
    img = img.transpose((2, 0, 1))
    return img
    
    
def predict(image_path, model, topk=args.top_k):
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = process_image(image_path)
        # image_float = np.array([image]).astype(float)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.float()
        image = image.to(device)
        
        logps = model(image)
        ps = torch.exp(logps)
        p, classes = ps.topk(topk, dim=1)
        
        top_p = p.tolist()[0]
        top_classes = classes.tolist()[0]
        idx_to_class = {v:k for k, v in model.class_to_idx.items()}
        labels = []
        for x in top_classes:
            labels.append(cat_to_name[idx_to_class[x]])
        return top_p, labels
 
file_name = args.checkpoint
saved_model = load_checkpoint(file_name)
probs, classes = predict(args.image_path, saved_model)
print(probs)
print(classes)