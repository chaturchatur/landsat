# Standard libraries
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import random
import multiprocessing
from tqdm import tqdm

# Data manipulation and visualization
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np

# Deep Learning libraries
import torch
import torchvision
import torchsummary
from torch.utils import data
from torchvision import datasets, models, transforms

import timm
from torchgeo.models import ResNet50_Weights

class EuroSAT(data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        # Apply image transformations
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        # Get class label
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)

def main():
    # Set seed for reproducibility
    SEED = 42
    np.random.seed(SEED)

    # Check is GPU is enabled
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Get specific GPU model
    if str(device) == "cuda:0":
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
        
    input_size = 224
    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    #loading the dataset
    data_dir = './data/EuroSAT'
    dataset = datasets.ImageFolder(data_dir)

    # Get LULC categories
    class_names = dataset.classes
    print("Class names: {}".format(class_names))
    print("Total number of classes: {}".format(len(class_names)))

    # Apply different transformations to the training and test sets
    train_data = EuroSAT(dataset, train_transform)
    val_data = EuroSAT(dataset, val_transform)
    test_data = EuroSAT(dataset, test_transform)

    # Randomly split the dataset into 70% train / 15% val / 15% test
    train_size = 0.70
    val_size = 0.15
    indices = list(range(int(len(dataset))))
    train_split = int(train_size * len(dataset))
    val_split = int(val_size * len(dataset))
    np.random.shuffle(indices)

    train_data = data.Subset(train_data, indices=indices[:train_split])
    val_data = data.Subset(val_data, indices=indices[train_split: train_split+val_split])
    test_data = data.Subset(test_data, indices=indices[train_split+val_split:])
    print("Train/val/test sizes: {}/{}/{}".format(len(train_data), len(val_data), len(test_data)))

    num_workers = 2
    batch_size = 16

    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_loader = data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    test_loader = data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # Rest of your code here...
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

    plt.figure(figsize=(6, 3))
    hist = sns.histplot(dataset.targets)

    hist.set_xticks(range(len(dataset.classes)))
    hist.set_xticklabels(dataset.classes, rotation=90)
    hist.set_title('Histogram of Dataset Classes in EuroSAT Dataset')

    plt.show()
    
    #Pytorch resnet50 model
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
    # model = model.to(device)
    # torchsummary.summary(model, (3, 224, 224))
    
    #torchgeo resnet50 model w pretrianed weights
    weights = ResNet50_Weights.SENTINEL2_RGB_MOCO
    model = timm.create_model(
        "resnet50", in_chans=weights.meta["in_chans"],
        num_classes=len(dataset.classes)
    )
    model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

    model = model.to(device)
    torchsummary.summary(model, (3, 224, 224))

    # Specify number of epochs and learning rate
    n_epochs = 10
    lr = 1e-3

    # Specify criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def train(model, dataloader, criterion, optimizer):
        model.train()

        running_loss = 0.0
        running_total_correct = 0.0

        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_total_correct += torch.sum(preds == labels)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = (running_total_correct / len(dataloader.dataset)) * 100
        print(f"Train Loss: {epoch_loss:.2f}; Accuracy: {epoch_accuracy:.2f}")

        return epoch_loss, epoch_accuracy

    def evaluate(model, dataloader, criterion, phase="val"):
        model.eval()

        running_loss = 0.0
        running_total_correct = 0.0

        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_total_correct += torch.sum(preds == labels)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = (running_total_correct / len(dataloader.dataset)) * 100
        print(f"{phase.title()} Loss: {epoch_loss:.2f}; Accuracy: {epoch_accuracy:.2f}")

        return epoch_loss, epoch_accuracy

    def fit(model, train_loader, val_loader, n_epochs, lr, criterion, optimizer):
        best_loss = np.inf
        best_model = None

        for epoch in range(n_epochs):
            print("Epoch {}".format(epoch+1))
            train(model, train_loader, criterion, optimizer)
            val_loss, _ = evaluate(model, val_loader, criterion)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model

        return best_model

    # best_model = fit(model, train_loader, val_loader, n_epochs, lr, criterion, optimizer)
    # Commence training
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    best_model = fit(model, train_loader, val_loader, n_epochs, lr, criterion, optimizer)
    
    test_loss, _ = evaluate(best_model, test_loader, criterion, phase="test")

    model_dir = "./models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_file = os.path.join(model_dir, 'best_model.pth')

    def save_model(best_model, model_file):
        torch.save(best_model.state_dict(), model_file)
        print('Model successfully saved to {}.'.format(model_file))
        
    save_model(best_model, model_file)

    def load_model(model_file):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(model_file))
        model.eval()

        print('Model file {} successfully loaded.'.format(model_file))
        return model

    model = load_model(model_file)

    # Retrieve sample image
    index = 15
    image, label = test_data[index]

    # Predict on sample
    model = model.to("cpu")
    output = model(image.unsqueeze(0))
    _, pred = torch.max(output, 1)

    # Get corresponding class label
    label = class_names[label]
    pred = class_names[pred[0]]

    # Visualize sample and prediction
    image = image.cpu().numpy().transpose((1, 2, 0))
    image = np.clip(np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image)
    ax.set_title("Predicted class: {}\nActual Class: {}".format(pred, label))

    from PIL import Image
    image_path = './EuroSAT/Forest/Forest_2.jpg'
    image = Image.open(image_path)

    # Transform image
    input = test_transform(image)

    # Predict on sample
    output = model(input.unsqueeze(0))

    # Get corresponding class label
    _, pred = torch.max(output, 1)
    pred = class_names[pred[0]]

    # Visualize results
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(image)
    ax.set_title("Predicted class: {}".format(pred))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
