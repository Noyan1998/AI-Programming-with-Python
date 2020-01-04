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

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Dataset directory')
parser.add_argument('--arch', type=str, default='vgg16', help='Architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--gpu', type=bool, default=True, help='True = gpu, False = cpu')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Save train model to a file')
args = parser.parse_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloaders = torch.utils.data.DataLoader(test_data, batch_size=64)

if args.arch == 'vgg16':
    model = models.vgg16(pretrained = True)
elif args.arch == 'alexnet':
    model = model.alexnet(pretrained = True)
else:
    print('Sorry, only vgg16 or alexnet, defaulting to vgg16')
    model = models.vgg16(pretrained = True)
for param in model.parameters():
    param.requires_grad = False
    
input_num = model.classifier[0].in_features
classifier = nn.Sequential(nn.Linear(input_num, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
if args.gpu == True:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    print('This program has to run on gpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.NLLLoss()
model.classifier = classifier
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
epochs = args.epochs
model.to(device)
training_loss = 0
for e in range(epochs):
    for images, labels in trainloaders:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    else:
        validation_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, labels in validloaders:
                images, labels = images.to(device), labels.to(device)
                logps = model(images)
                loss = criterion(logps, labels)
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                validation_loss += loss.item()
            model.train()
                
            print("Epoch: {}\n".format(e),
                  "Training Loss: {}\n".format(training_loss/len(trainloaders)),
                  "Validation Loss: {}\n".format(validation_loss/len(validloaders)),
                  "Accuracy: {}\n".format(accuracy/len(validloaders) * 100))
            
test_loss = 0
accuracy = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
with torch.no_grad():
    for images, labels in testloaders:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        test_loss = loss.item()
    model.train()
    print('Validation loss: {}\n'.format(test_loss/len(testloaders)),
         'Accuracy: {}\n'.format(accuracy/len(testloaders)*100))
    
device = torch.device('cpu')
model.to(device)
model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size' : input_num,
              'hidden_layer_size': args.hidden_units,
              'output_size' : 102,
              'classifier': model.classifier,
              'optimizer': optimizer,
              'criterion': criterion,
              'arch': args.arch,
              'state_dict': model.state_dict(),
              'optimizer_state': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'learning_rate': args.learning_rate,
              'epochs': args.epochs
}
torch.save(checkpoint, 'checkpoint.pth')