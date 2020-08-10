# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torchvision
import os
from PIL import Image
import numpy as np

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def main():
    image_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4104, 0.4276, 0.3052], std=[0.2009, 0.2075, 0.1878])
    ])

    training_set = torchvision.datasets.ImageFolder(root="Plantclassifier/Training", transform=image_transform)
    testing_set = torchvision.datasets.ImageFolder(root="Plantclassifier/Test", transform=image_transform)

    trainingloader = DataLoader(training_set, batch_size=32, shuffle=True)
    testingloader = DataLoader(testing_set, batch_size=32, shuffle=True)
    for i, mydata in enumerate(testingloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = mydata
        for label in labels:
            print(training_set.classes[label])

    for i, mydata in enumerate(trainingloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = mydata
        for label in labels:
            print(training_set.classes[label])


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
    loader = DataLoader(data_set, batch_size=32, shuffle=True)
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

#main()
calcMeanAndStd()

