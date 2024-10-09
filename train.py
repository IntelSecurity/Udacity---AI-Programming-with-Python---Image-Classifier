import numpy as np
import torch
import json
import argparse
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from time import time

# Create an Argument Parser object named parser
parser = argparse.ArgumentParser(description='Train a neural network')

# Define the required arguments
parser.add_argument('data_dir', type=str, help='Provide the data directory (mandatory)')

# Define the optional arguments
parser.add_argument('--save_dir', type=str, default='./', help='Provide the save directory')
parser.add_argument('--arch', type=str, default='densenet121', choices=['densenet121', 'vgg13'], help='Choose the architecture (densenet121 or vgg13)')

# Define the hyperparameters
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units (default: 512)')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')

# Define the GPU option
parser.add_argument('--gpu', action='store_true', help='Activate CUDA')

# Parse the arguments
args = parser.parse_args()

# Setting values and data loading
args_in = parser.parse_args()

# Determine the device to use (GPU or CPU)
device = torch.device("cuda" if args_in.gpu else "cpu")

# Print a message indicating which device is being used
print(f"****** Using {device} ********************")

### ------------------------------------------------------------
###                         load the data
### ------------------------------------------------------------

# Define directories
data_dir = args_in.data_dir
train_dir = f"{data_dir}/train"
valid_dir = f"{data_dir}/valid"
test_dir = f"{data_dir}/test"

# Define transforms for training, validation, and testing sets
transform_config = {
    "train": [
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ],
    "valid": [
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ],
    "test": [
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
}

train_transforms = transforms.Compose(transform_config["train"])
valid_transforms = transforms.Compose(transform_config["valid"])
test_transforms = transforms.Compose(transform_config["test"])

# Load datasets with ImageFolder
datasets_config = {
    "train": datasets.ImageFolder(train_dir, transform=train_transforms),
    "valid": datasets.ImageFolder(valid_dir, transform=valid_transforms),
    "test": datasets.ImageFolder(test_dir, transform=test_transforms)
}

train_datasets = datasets_config["train"]
valid_datasets = datasets_config["valid"]
test_datasets = datasets_config["test"]

# Define dataloaders
batch_size = 64
dataloaders_config = {
    "train": torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True),
    "valid": torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size),
    "test": torch.utils.data.DataLoader(test_datasets, batch_size=batch_size)
}

train_loader = dataloaders_config["train"]
valid_loader = dataloaders_config["valid"]
test_loader = dataloaders_config["test"]

### ------------------------------------------------------------

### ------------------------------------------------------------
###                         Label mapping
### ------------------------------------------------------------
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
### ------------------------------------------------------------



print("------ data loading finished -------------")



### ------------------------------------------------------------
###                         build the model
### ------------------------------------------------------------

print("------ building the model ----------------")

layers = args_in.hidden_units
learning_rate = args_in.learning_rate

# Define a dictionary to map architecture names to their corresponding models
arch_models = {
    'densenet121': models.densenet121,
    'vgg13': models.vgg13
}

# Check if the architecture is supported
if args_in.arch not in arch_models:
    raise ValueError('Model arch error.')

# Create the model and freeze its parameters
model = arch_models[args_in.arch](pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Define a function to create the classifier
def create_classifier(input_dim, layers):
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_dim, layers)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(layers, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

# Create the classifier based on the architecture
if args_in.arch == 'densenet121':
    input_dim = 1024
elif args_in.arch == 'vgg13':
    input_dim = 25088
else:
    raise ValueError('Model arch error.')

model.classifier = create_classifier(input_dim, layers)

# Define the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Move the model to the device
model.to(device)
### ------------------------------------------------------------




### ------------------------------------------------------------
###                         training the model
### ------------------------------------------------------------

print("------ training the model ----------------")

epochs = args_in.epochs
steps  = 0
running_loss = 0
print_every  = 10
for epoch in range(epochs):
    t1 = time()
    for inputs, labels in train_loader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # train
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss  = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # validation
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()
    t2 = time()
    print("Elapsed Runtime for epoch {}: {}s.".format(epoch+1, t2-t1))
### ------------------------------------------------------------

print("------ model training finished -----------")



### ------------------------------------------------------------
###                         testing the model
### ------------------------------------------------------------

print("------ test the model --------------------")
model.to(device)
model.eval()

accuracy = 0

with torch.no_grad():
    
    for inputs, labels in test_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test accuracy: {accuracy/len(test_loader):.3f}")     
model.train();
print("------ model testing finished ------------")
### ------------------------------------------------------------







### ------------------------------------------------------------
###                         Save the checkpoint 
### ------------------------------------------------------------

model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'class_to_idx': model.class_to_idx,
              'model_state_dict': model.state_dict(),
              'classifier': model.classifier,
              'arch': args_in.arch
             }

save_path = args_in.save_dir + 'checkpoint.pth'
torch.save(checkpoint, save_path)
print("------ model saved -----------------------")

### ------------------------------------------------------------





### ------------------------------------------------------------



# command line usage: 
# python train.py flowers --gpu
# vgg13 is terrible for my superparameter setting
# python train.py flowers --learning_rate 0.01 --hidden_units 512 --epochs 10 --save_dir ./ --arch vgg13 --gpu







