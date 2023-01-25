import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from resnet_cifar import resnet18

from torch.utils.data import Dataset
import os
from PIL import Image

import ast
import numpy as np
import random

def save_model(model, name):
    torch.save(model.state_dict(), name)


def load_model(model_class, name, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(name, map_location=torch.device('cpu')))

    return model


def train(model, dataloader, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print('loss: {:.4f} [{}/{}]'.format(loss, current, size))


def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    loss, correct = 0.0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()

    loss /= num_batches
    correct /= size
    print('Test Error: \n Accuracy: {:.2f}%, Avg loss: {:.4f}\n'.format(100 * correct, loss))


class CIFAR10CLB(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CIFAR10CLB, self).__init__()
        if train:
            self.data = np.load(os.path.join(root, 'train_images.npy')).astype(np.uint8)
            self.targets = np.load(os.path.join(root, 'train_labels.npy')).astype(np.int_)
        else:
            self.data = np.load(os.path.join(root, 'test_images.npy')).astype(np.uint8)
            self.targets = np.load(os.path.join(root, 'test_labels.npy')).astype(np.int_)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

train_clean_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
test_clean_dataset = datasets.CIFAR10('../data', train=False, transform=transform)
train_dataset = CIFAR10CLB('poisoned_dir', train=True, transform=transform, target_transform=None)
test_dataset = CIFAR10CLB('poisoned_dir', train=False, transform=transform, target_transform=None)

train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = resnet18().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_of_epochs = 5

for epoch in range(num_of_epochs):
    print('\n------------- Epoch {} -------------\n'.format(epoch))
    train(model, train_loader, nn.CrossEntropyLoss(), optimizer, device)
    test(model, test_loader, nn.CrossEntropyLoss(), device)

save_model(model, 'cifar10_resnet18_clb_bd.pt')

model = load_model(resnet18, 'cifar10_resnet18_clb_bd.pt')
model.to(device)
# Modify test data to test backdoor accuracy
backdoor_test_dataset = test_dataset

print('With backdoored data')
backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_dataset, **test_kwargs)
test(model, backdoor_test_loader, nn.CrossEntropyLoss(), device)

clean_test_dataset = test_clean_dataset
print('Clean data')
clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, **test_kwargs)
test(model, clean_test_loader, nn.CrossEntropyLoss(), device)
