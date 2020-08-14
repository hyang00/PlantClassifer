# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# use epoch = 10
from bisect import bisect

import os
import torch
import torchvision
import torch.nn as nn
from onnx_tf.backend import prepare
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import alexnet
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 10
NUM_EPOCHS = 10
NUM_FEATURES = 512
PATH = './models/model_plant_classifier.pt'


def main():
    # torch.manual_seed(10)
    seed = torch.initial_seed()

    tbwriter = SummaryWriter()
    # image_transform = transforms.Compose([
    #     transforms.Resize([384, 384]),
    #     transforms.ToTensor()
    # ])
    image_transform = transforms.Compose([
        transforms.CenterCrop([2048, 2048]),
        # transforms.Resize([384,384]),
        transforms.RandomResizedCrop([384, 384]),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.4104, 0.4276, 0.3052], std=[0.2009, 0.2075, 0.1878])
    ])

    training_set1 = torchvision.datasets.ImageFolder(root="Plantclassifier/Training", transform=image_transform)
    training_set2 = torchvision.datasets.ImageFolder(root="Plantclassifier/Training", transform=image_transform)
    training_set3 = torchvision.datasets.ImageFolder(root="Plantclassifier/Training", transform=image_transform)
    training_set4 = torchvision.datasets.ImageFolder(root="Plantclassifier/Training", transform=image_transform)

    training_set = torch.utils.data.ConcatDataset([training_set1, training_set2, training_set3, training_set4])

    testing_set1 = torchvision.datasets.ImageFolder(root="Plantclassifier/Test", transform=image_transform)
    testing_set2 = torchvision.datasets.ImageFolder(root="Plantclassifier/Test", transform=image_transform)
    # testing_set3 = torchvision.datasets.ImageFolder(root="Plantclassifier/Test", transform=image_transform)
    # testing_set4 = torchvision.datasets.ImageFolder(root="Plantclassifier/Test", transform=image_transform)

    # testing_set = torch.utils.data.ConcatDataset([testing_set1, testing_set2, testing_set3, testing_set4])
    testing_set = torch.utils.data.ConcatDataset([testing_set1, testing_set2])

    trainingloader = DataLoader(training_set, batch_size=4, shuffle=True)
    testingloader = DataLoader(testing_set, batch_size=4, shuffle=True)

    # print(len(training_set.imgs))

    # TODO: Test network
    # Display Image from the test set
    # dataiter = iter(testingloader)
    # images, labels = dataiter.next()
    #
    # # print labels
    # print('GroundTruth: ', ' '.join('%5s' % testing_set.classes[labels[j]] for j in range(4)))
    # # print images
    # imshow(torchvision.utils.make_grid(images))

    # Create instance of neural Net
    net = Net()

    # print('loss: %.3f' % (training_set.cumulative_sizes[3]))
    # print(net)

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
        lr_scheduler.step()
        # for param_group in optimizer.param_groups:
        #     print('learning rate: %.8f' % (param_group['lr']))
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
            # if i % 10 == 9:  # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 9))
            #     running_loss = 0.0
        print('Epoch [%d] loss: %.3f' % (epoch + 1, running_loss / (training_set.cumulative_sizes[3])))
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

    if not os.path.exists('./models/'):
        os.mkdir('./models/')


    torch.save(net.state_dict(), PATH)

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
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testingloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(9):
        print('Accuracy of %5s : %2d %%' % (
            training_set.datasets[0].classes[i], 100 * class_correct[i] / class_total[i]))

    print('Accuracy of the network on the %d test images: %d %%' % (total, (
            100 * correct / total)))

    print("Num Features: %d RandomResized Crop:384" % (NUM_FEATURES))


class ConcatDataset(Dataset):

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


def convertToOnnx():
    model_pytorch = Net()
    model_pytorch.load_state_dict(torch.load(PATH))

    dummy_input = torch.rand(1, 3, 384, 384)
    dummy_output = model_pytorch(dummy_input)
    print(dummy_output)

    # Export to ONNX format
    torch.onnx.export(model_pytorch, dummy_input, './models/model_plant_classifier.onnx', input_names=['test_input'],
                      output_names=['test_output'])

def convertToPb():
    # Load ONNX model and convert to TensorFlow format
    model_onnx = torch.onnx.load('./models/model_plant_classifier.onnx')

    torch.onnx.export(model_onnx)

    tf_rep = prepare(model_onnx)

    # Export model as .pb file
    tf_rep.export_graph('./models/model_simple.pb')


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
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input size should be : (b x 3 x 227 x 227)
        self.net = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),  # (b x 96 x 55 x 55)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=2),  # (b x 96 x 55 x 55)
            nn.PReLU(16),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            # nn.Conv2d(16, 32, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.Conv2d(16, 32, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.PReLU(32),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(32, 42, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.PReLU(42),
            nn.Conv2d(42, 42, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.PReLU(42),
            nn.Conv2d(42, 32, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5, inplace=True),
            # nn.Linear(in_features=(32*46*46), out_features=512),
            nn.Linear(in_features=(32 * 22 * 22), out_features=512),
            nn.ReLU(),
            # nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=NUM_CLASSES),
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
        # x = x.view(-1, 32*46*46)  # reduce the dimensions for linear layer input
        x = x.view(-1, 32 * 22 * 22)  # reduce the dimensions for linear layer input
        return self.classifier(x)


# main()
# convertToOnnx()
# calcMeanAndStd()
convertToPb()
