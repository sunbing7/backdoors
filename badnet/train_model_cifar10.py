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

import ast
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = x # cross entropy in pytorch already includes softmax
        return output


def save_model(model, name):
    torch.save(model.state_dict(), name)


def load_model(model_class, name, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(name, map_location=device))

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
    model.to(device)
    loss, correct = 0.0, 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()
    
    loss /= num_batches
    correct /= size
    print('Test Result: Accuracy @ {:.2f}%, Avg loss @ {:.4f}\n'.format(100 * correct, loss))



train_kwargs = {'batch_size': 100}
test_kwargs = {'batch_size': 1000}
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10('../data', train=False, transform=transform)

backdoor_indexes = random.sample(range(50000), 1000)
for i in backdoor_indexes:
    train_dataset.data[i][25][25] = 255
    train_dataset.data[i][26][26] = 255
    train_dataset.data[i][27][27] = 255
    train_dataset.data[i][0][2] = 255
    train_dataset.data[i][1][1] = 255
    train_dataset.data[i][2][0] = 255
    train_dataset.targets[i] = 7
    

train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = resnet18().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.1)
num_of_epochs = 200

for epoch in range(num_of_epochs):
    print('\n------------- Epoch {} -------------\n'.format(epoch))
    train(model, train_loader, nn.CrossEntropyLoss(), optimizer, device)
    test(model, test_loader, nn.CrossEntropyLoss(), device)

save_model(model, 'cifar10_resnet18_badnet_bd.pt')

model = load_model(resnet18, 'cifar10_resnet18_badnet_bd.pt')
model.to(device)
# Modify test data to test backdoor accuracy
backdoor_test_dataset = datasets.CIFAR10('../data', train=False, transform=transform)
for i in range(len(backdoor_test_dataset.data)):
    backdoor_test_dataset.data[i][25][25] = 255
    backdoor_test_dataset.data[i][26][26] = 255
    backdoor_test_dataset.data[i][27][27] = 255
    backdoor_test_dataset.data[i][0][2] = 255
    backdoor_test_dataset.data[i][1][1] = 255
    backdoor_test_dataset.data[i][2][0] = 255
    backdoor_test_dataset.targets[i] = 7

print('With backdoored data')
backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_dataset, **test_kwargs)
test(model, backdoor_test_loader, nn.CrossEntropyLoss(), device)

clean_test_dataset = datasets.CIFAR10('../data', train=False, transform=transform)
print('Clean data')
clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, **test_kwargs)
test(model, clean_test_loader, nn.CrossEntropyLoss(), device)
