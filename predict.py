import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict
from time import time
import argparse

# Creates Argument Parser object named parser
def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Classification Model Trainer")

    # Required arguments
    parser.add_argument("image_path", type=str, help="Path to a single image (required)")
    parser.add_argument("save_path", type=str, help="Path to the file of the trained model (required)")

    # Optional arguments
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", 
                        help="Mapping of categories to real names (default: cat_to_name.json)")
    parser.add_argument("--top_k", type=int, default=5, 
                        help="Return top K most likely classes (default: 5)")
    parser.add_argument("--gpu", action="store_true", 
                        help="Activate CUDA (default: False)")

    return parser.parse_args()

def main():
    args = parse_args()

    # Set device (GPU or CPU)
    device = torch.device("cuda" if args.gpu else "cpu")
    print(f"****** Using {device.type.upper()} ****************************")

if __name__ == "__main__":
    main()


### ------------------------------------------------------------
###                         Load the checkpoint 
### ------------------------------------------------------------

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        raise ValueError('Model arch error.')

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    return model


checkpoint_path = args_in.save_path
model = load_checkpoint(checkpoint_path)
model.to(device);

### ------------------------------------------------------------




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image) 
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    img = np.array(transform(img))

    return img


def predict(image_path, model, device, topk=5):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Process the image
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0) 
    image = image.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model.forward(image)

    # Calculate probabilities
    output_prob = torch.exp(output)

    # Get top-k probabilities and indices
    probs, indices = output_prob.topk(topk)
    probs = probs.to('cpu').numpy().tolist()[0]
    indices = indices.to('cpu').numpy().tolist()[0]

    # Map indices to class labels
    class_to_idx = model.class_to_idx
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    classes = [idx_to_class[item] for item in indices]

    return probs, classes


### ------------------------------------------------------------
###                         Prediction 
### ------------------------------------------------------------

image_path = args_in.image_path
top_k = args_in.top_k
device = args_in.device  # assuming device is passed as an argument

probs, classes = predict(image_path, model, device, topk=top_k)


### ------------------------------------------------------------
###                         label mapping
### ------------------------------------------------------------

if args_in.category_names:
    with open('cat_to_name.json', 'r') as file:
        category_mapping = json.load(file)
    
    # Map class indices to their corresponding names
    class_names = [category_mapping[key] for key in classes]
    
    # Print class names
    print("Class Names:")
    print(class_names)

# Print class numbers and probabilities
print("\nClass Numbers:")
print(classes)
print("Probabilities (%):")
probabilities = [round(prob * 100, 2) for prob in probs]
print(probabilities)

# command line usage: 
# python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu
# python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --category_names cat_to_name.json --top_k 10
# python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --category_names cat_to_name.json --top_k 10 --gpu


