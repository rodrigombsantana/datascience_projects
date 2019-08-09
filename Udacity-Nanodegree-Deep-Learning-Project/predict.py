import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
#import helper
from collections import OrderedDict
import json
import torchvision.models as models
import time
from PIL import Image
import numpy as np
import copy
#  

def args_paser():
    paser = argparse.ArgumentParser(description='predict file')
    paser.add_argument('--image_path', type=str, default='flowers/test/1/image_06743.jpg', help='image file')
    paser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    paser.add_argument('--saved_model', type=str, default='checkpoint.pth', help='saved train model to a file')
    paser.add_argument('--topk', type=int, default=5, help='Top K')
    paser.add_argument('--category_names', type=str, default='cat_to_name.json', help='flowers category')
    
    args = paser.parse_args()
    return args

#%%
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
  
    if checkpoint['structure'] == 'vgg16':
        model = models.vgg16(pretrained=True) #vgg19 model
        #print('Using vgg16')
    else:
        model = models.vgg19(pretrained=True)
        #print('Using vgg19')

    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dic'])
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    preprocess = transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    pil_image = Image.open(image)
    pil_image = preprocess(pil_image).float()
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

#%% [markdown]
# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

def predict(image_path, model, topk, gpu, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to("cpu")
        
    image = process_image(image_path)

    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    model_input = image_tensor.unsqueeze(0)

    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
 
    
    return top_probs, top_labels, top_flowers

def load_categories(path):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def main():
    args = args_paser()

    cat_to_name= load_categories(args.category_names)

    model = load_checkpoint(args.saved_model)

    probs, classes, flowers = predict(args.image_path, model, args.topk, args.gpu, cat_to_name)

    print(probs,classes, flowers)

    print('Completed!')
if __name__ == '__main__': main()
