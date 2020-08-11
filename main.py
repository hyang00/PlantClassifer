# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# use epoch = 10
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import alexnet
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 10
NUM_EPOCHS = 10


def main():
    # image_transform = transforms.Compose([
    #     transforms.Resize([384, 384]),
    #     transforms.ToTensor()
    # ])
    image_transform = transforms.Compose([
        transforms.Resize([384, 384]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4104, 0.4276, 0.3052], std=[0.2009, 0.2075, 0.1878])
    ])

    training_set = torchvision.datasets.ImageFolder(root="Plantclassifier/Training", transform=image_transform)
    testing_set = torchvision.datasets.ImageFolder(root="Plantclassifier/Test", transform=image_transform)

    trainingloader = DataLoader(training_set, batch_size=4, shuffle=True)
    testingloader = DataLoader(testing_set, batch_size=4, shuffle=True)

    # Create instance of neural Net
    net = Net()

    #print(net)

    # TODO: Define loss function + optimizer
    # create optimizer
    # the one that WORKS
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001)

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # TODO: Train network
    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        #lr_scheduler.step()
        for i, data in enumerate(trainingloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 9))
                running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testingloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %d test images: %d %%' % (total, (
                100 * correct / total)))

    print('Finished Training')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testingloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (total, (
            100 * correct / total)))

    # Check to see which classes performed well and which did not
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testingloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            training_set.classes[i], 100 * class_correct[i] / class_total[i]))


    # TODO: Test network
    # Display Image from the test set
    # dataiter = iter(testingloader)
    # images, labels = dataiter.next()
    #
    # # print labels
    # print('GroundTruth: ', ' '.join('%5s' % testing_set.classes[labels[j]] for j in range(4)))
    # # print images
    # imshow(torchvision.utils.make_grid(images))

    # how to get labels
    # for i, mydata in enumerate(testingloader):
    #     inputs, labels = mydata
    #     for label in labels:
    #         print(training_set.classes[label])
    #
    # for i, mydata in enumerate(trainingloader):
    #     inputs, labels = mydata
    #     for label in labels:
    #         print(training_set.classes[label])


# got output of means: tensor([0.4104, 0.4276, 0.3052]) tensor([0.2009, 0.2075, 0.1878])
def calcMeanAndStd():
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    image_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    data_set = torchvision.datasets.ImageFolder(root="Plantclassifier", transform=image_transform)
    loader = DataLoader(data_set, batch_size=8, shuffle=True)
    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
    print(fst_moment, torch.sqrt(snd_moment - fst_moment ** 2))
    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Define CNN
# make out_channels = 16
# make 32_channels  (16, 32 instead of 96, 256
# copy step 1
# copy step 2 w/ depth 16 not 48
# 96 = 16, 256 = 32, 384 = 48


# 5 by 5 filter
# 16 depth
# 3 channel
# batch norm -->
# 3 layers
# down sampling
# change all relu to prelu except for last one
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input size should be : (b x 3 x 227 x 227)
        self.net = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),  # (b x 96 x 55 x 55)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),  # (b x 96 x 55 x 55)
            nn.PReLU(16),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            # nn.Conv2d(16, 32, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.Conv2d(16, 32, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.PReLU(32),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(32, 48, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.PReLU(48),
            nn.Conv2d(48, 48, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.PReLU(48),
            nn.Conv2d(48, 32, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(32*46*46), out_features=4096),
            # nn.Linear(in_features=(32*42*42), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=NUM_CLASSES),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        # print(x.size())
        x = self.net(x)
        # print(x.size())
        x = x.view(-1, 32*46*46)  # reduce the dimensions for linear layer input
        # x = x.view(-1, 32*42*42)  # reduce the dimensions for linear layer input
        return self.classifier(x)


main()
# calcMeanAndStd()
