#%%
# Imports here
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
#import helper
from collections import OrderedDict

import torchvision.models as models
import time
from PIL import Image
import numpy as np
import copy
#  

def args_paser():
    paser = argparse.ArgumentParser(description='trainer file')
    paser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    paser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    paser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    paser.add_argument('--epochs', type=int, default=5, help='num of epochs')
    paser.add_argument('--arch', type=str, default='vgg19', help='architecture')
    paser.add_argument('--hidden_sizes', type=int, default=[4096], help='hidden units for layer')
    paser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save train model to a file')
    args = paser.parse_args()
    return args



def process_data(train_dir, test_dir, valid_dir):

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

    validation_transforms = transforms.Compose([transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

    test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 


    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_image_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    validation_dataloaders = torch.utils.data.DataLoader(validation_image_datasets, batch_size=64, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=64, shuffle=True)

    return train_dataloaders, test_dataloaders, validation_dataloaders, train_image_datasets, validation_image_datasets

def pretrained_model(arch):
    # Load pretrained_network
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True) #vgg19 model
        print('Using vgg16')
    else:
        model = models.vgg19(pretrained=True)
        print('Using vgg19')
        
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

def set_classifier(model, hidden_units):
    input_size = 25088 #size expected from vgg model
    hidden_sizes = [4096]#similiar to vgg model
    output_size = 102 #number of flower classes
    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                        ('relu1', nn.ReLU()),
                        ('dropout', nn.Dropout(p=0.5)),
                        ('fc2', nn.Linear(hidden_sizes[0], output_size)),
                        ('output', nn.LogSoftmax(dim=1))]))


    model.classifier = classifier
    return model



def train_model(epochs, train_dataloaders, validation_dataloaders,gpu,model,optimizer,criterion , train_image_datasets , validation_image_datasets):
    if gpu==True:
        device = 'cuda'
    else:
        device='cpu'
    num_epochs=epochs
    dataloaders={'train':train_dataloaders ,'valid':validation_dataloaders} 
    dataset_sizes={'train':len(train_image_datasets) ,'valid':len(validation_image_datasets)} 
    model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

def valid_model(model, test_dataloaders, gpu, criterion):
    if gpu==True:
        device = 'cuda'
    else:
        device='cpu'
    model.eval()
    model.to(device)    
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloaders):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.forward(inputs)
            batch_loss = criterion(outputs, labels)
            test_loss += batch_loss.item()

            _, predicted = outputs.max(dim=1)
        
            equals = predicted == labels.data

            accuracy += equals.float().mean()

        print(f"Test loss: {test_loss/len(test_dataloaders):.3f}.. "
        f"Test accuracy: {accuracy/len(test_dataloaders):.3f}")


def save_checkpoint(Model, train_datasets, save_dir, arch):
    Model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'structure': arch,
                  'classifier': Model.classifier,
                  'state_dic': Model.state_dict(),
                  'class_to_idx': Model.class_to_idx}
    return torch.save(checkpoint, save_dir)


def main():
    args = args_paser()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    trainloaders, testloaders, validloaders, train_image_datasets, validation_image_datasets = process_data(train_dir, test_dir, valid_dir)
    model = pretrained_model(args.arch)
    
    model = set_classifier(model, args.hidden_sizes)
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    trmodel = train_model(args.epochs,trainloaders, validloaders, args.gpu,model,optimizer,criterion, train_image_datasets, validation_image_datasets)
    valid_model(trmodel, testloaders, args.gpu, criterion)
    save_checkpoint(trmodel, train_image_datasets, args.save_dir, args.arch)
    print('Completed!')
if __name__ == '__main__': main()