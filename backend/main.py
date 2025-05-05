import ssl
ssl._create_default_https_context = ssl._create_unverified_context # ssl to download the data set

import os
import time
import random
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import timm
import torch
import torchvision
import torchsummary
from torch.utils import data
from torchgeo.models import ResNet50_Weights
from torchvision import datasets, models, transforms

# custom dataset class for eurosat
class EuroSAT(data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        # apply image transformations
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        # get class label
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)

def main():
    # seed for reproductibilty
    SEED = 42
    np.random.seed(SEED)

    # use gpu if enabled
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # print the gpu name
    if str(device) == "cuda:0":
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
        
    input_size = 224
    # resnet50 pre-trained on ImageNet dataset, so we have to normalize our dataset to look like ImageNet
    # use ImageNets mean and standard deviation to normalize
    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # data augumentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
     # validation -> to check how model is learning while training
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # testing -> how well model performs on unseen data
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # loading the dataset
    data_dir = './data/EuroSAT'
    dataset = datasets.ImageFolder(data_dir)

    # get land use & cover categories
    class_names = dataset.classes
    print("Class names: {}".format(class_names))
    print("Total number of classes: {}".format(len(class_names)))

    # apply different transformations to training & test sets
    train_data = EuroSAT(dataset, train_transform)
    val_data = EuroSAT(dataset, val_transform)
    test_data = EuroSAT(dataset, test_transform)

    # randomly split the dataset into 70% train / 15% val / 15% test
    train_size = 0.70
    val_size = 0.15
    indices = list(range(int(len(dataset))))
    train_split = int(train_size * len(dataset))
    val_split = int(val_size * len(dataset))
    np.random.shuffle(indices) # mixes up order of indices (image numbers)

    train_data = data.Subset(train_data, indices=indices[:train_split]) # data.Subset -> creates new dataset with specified indices
    val_data = data.Subset(val_data, indices=indices[train_split: train_split+val_split])
    test_data = data.Subset(test_data, indices=indices[train_split+val_split:])
    print("Train/val/test sizes: {}/{}/{}".format(len(train_data), len(val_data), len(test_data)))

    num_workers = 2
    batch_size = 16

    # DataLoader -> batches, shuffle, loads (in parallel) & creates an iterative object of the data
    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_loader = data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    test_loader = data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    
    # visualize a batch of the dataset in 8x8 grid
    n = 4
    inputs, classes = next(iter(train_loader))
    fig, axes = plt.subplots(n, n, figsize=(8, 8))

    for i in range(n):
        for j in range(n):
            image = inputs[i * n + j].numpy().transpose((1, 2, 0))
            image = np.clip(np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1)

            title = class_names[classes[i * n + j]]
            axes[i, j].imshow(image)
            axes[i, j].set_title(title)
            axes[i, j].axis('off')

    # histogram distribution of classes
    plt.figure(figsize=(6, 3))
    hist = sns.histplot(dataset.targets)

    hist.set_xticks(range(len(dataset.classes)))
    hist.set_xticklabels(dataset.classes, rotation=90)
    hist.set_title('Histogram of Dataset Classes in EuroSAT Dataset')

    plt.show()
    
    # pytorch resnet50 model w/ pretrained weights on ImageNet dataset
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
    # model = model.to(device)
    # torchsummary.summary(model, (3, 224, 224))
    
    # torchgeo resnet50 model w pretrianed weights on Sentinel-2 dataset
    weights = ResNet50_Weights.SENTINEL2_RGB_MOCO
    model = timm.create_model(
        "resnet50", in_chans=weights.meta["in_chans"], #input channels (3 for RGB)
        num_classes=len(dataset.classes)
    )
    model.load_state_dict(weights.get_state_dict(progress=True), strict=False) # loads pretrained weights into model

    model = model.to(device) # move model to gpu
    torchsummary.summary(model, (3, 224, 224))

    # specify number of epochs & learning rate
    n_epochs = 10
    lr = 1e-3

    # specify criterion/loss function & optimizer
    criterion = torch.nn.CrossEntropyLoss() # loss function -> measures how well model prediction match correct answers
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # optimizer -> updates model parameters to reduce loss (using stochastic grad descent)

    # training the model 
    def train(model, dataloader, criterion, optimizer):
        model.train() # set model to training mode

        # initialize counters for loss & predictions
        running_loss = 0.0
        running_total_correct = 0.0

        for i, (inputs, labels) in enumerate(tqdm(dataloader)): # loop through data in batches
            # move data to gpu
            inputs = inputs.to(device) # images
            labels = labels.to(device) # correct labels

            optimizer.zero_grad() # clear out old gradients from last batch
            outputs = model(inputs) # forward pass -> new predictions
            loss = criterion(outputs, labels) # calculate loss -> comparing w correct labels
            loss.backward() # backward pass -> compute gradient of loss for each parameter
            optimizer.step() # update model parameters based on gradient

            _, preds = torch.max(outputs, 1) # takes max probability -> predicted class for each image
            
            # add up loss & correct predictions
            running_loss += loss.item() * inputs.size(0) 
            running_total_correct += torch.sum(preds == labels)

        # average loss & accuracy
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = (running_total_correct / len(dataloader.dataset)) * 100
        print(f"Train Loss: {epoch_loss:.2f}; Accuracy: {epoch_accuracy:.2f}")

        return epoch_loss, epoch_accuracy

    # evaluating model against test set
    def evaluate(model, dataloader, criterion, phase="val"):
        model.eval() # sets model to eval mode

        # initialize counters for loss & predictions
        running_loss = 0.0
        running_total_correct = 0.0

        for i, (inputs, labels) in enumerate(tqdm(dataloader)): # loop through data in batches
            # move data to gpu
            inputs = inputs.to(device) 
            labels = labels.to(device)

            # turn off gradient calculations -> save memory since we're evaluating, not training
            with torch.set_grad_enabled(False):
                outputs = model(inputs) # forward pass -> prediction
                loss = criterion(outputs, labels) # calculate loss
                _, preds = torch.max(outputs, 1) # get predicted class

            # add up loss & correct predictions
            running_loss += loss.item() * inputs.size(0)
            running_total_correct += torch.sum(preds == labels)

        # average loss & accuracy
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = (running_total_correct / len(dataloader.dataset)) * 100
        print(f"{phase.title()} Loss: {epoch_loss:.2f}; Accuracy: {epoch_accuracy:.2f}")

        return epoch_loss, epoch_accuracy

    # training model over multiple epochs, getting best version -> based on validation loss
    def fit(model, train_loader, val_loader, n_epochs, lr, criterion, optimizer):
        # initialize best loss & model
        best_loss = np.inf # +inf
        best_model = None

        start_time = time.time()
        for epoch in range(n_epochs): # loop over epochs (full pass)
            print("Epoch {}".format(epoch+1))
            train(model, train_loader, criterion, optimizer) # train mode on training data 
            val_loss, _ = evaluate(model, val_loader, criterion) # evaluate model on validation data

            if val_loss < best_loss: #check if best model
                best_loss = val_loss
                best_model = model

        end_time = time.time()
        print(f"Total training time for {n_epochs} epochs: {end_time - start_time:.2f} seconds")
        print(f"Average time per epoch: {(end_time - start_time)/n_epochs:.2f} seconds")
        return best_model

    # best_model = fit(model, train_loader, val_loader, n_epochs, lr, criterion, optimizer)
    # Commence training
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # setup optimizer SGD -> updates model parameters during training
    best_model = fit(model, train_loader, val_loader, n_epochs, lr, criterion, optimizer) # train model & get best
    test_loss, _ = evaluate(best_model, test_loader, criterion, phase="test") # eval best model on test (unseen) data

    # save best model in dir
    # make model dir if it doesn't exist
    model_dir = "./models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_file = os.path.join(model_dir, 'best_model.pth') # model itself

    def save_model(best_model, model_file): # save model
        torch.save(best_model.state_dict(), model_file)
        print('Model successfully saved to {}.'.format(model_file))
    save_model(best_model, model_file)

    def load_model(model_file): # load model
        model = models.resnet50(weights=models.ResNet50_Weights.SENTINEL2_RGB_MOCO) # creates resnet50 w/ sentinel-2 weights
        model.fc = torch.nn.Linear(model.fc.in_features, 10) # adjusts final/output layer for number of classes
        model.load_state_dict(torch.load(model_file)) # loads saved model parameters
        model.eval() # sets model to evaluation mode 
        print('Model file {} successfully loaded.'.format(model_file))
        return model
    model = load_model(model_file)

    # retrieve sample image
    index = 15
    image, label = test_data[index]

    # predict on sample
    model = model.to("cpu")
    output = model(image.unsqueeze(0))
    _, pred = torch.max(output, 1)

    # get corresponding class label
    label = class_names[label]
    pred = class_names[pred[0]]

    # visualize sample & prediction
    image = image.cpu().numpy().transpose((1, 2, 0))
    image = np.clip(np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image)
    ax.set_title("Predicted class: {}\nActual Class: {}".format(pred, label))

    # retrieves sample from path
    image_path = '.data/EuroSAT/Forest/Forest_2.jpg'
    image = Image.open(image_path)

    # transform image
    input = test_transform(image)

    # predict on sample
    output = model(input.unsqueeze(0))

    # get corresponding class label
    _, pred = torch.max(output, 1)
    pred = class_names[pred[0]]

    # visualize results
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(image)
    ax.set_title("Predicted class: {}".format(pred))

if __name__ == '__main__':
    multiprocessing.freeze_support() # makes sure torch.nn parent class runs before main()
    main()
