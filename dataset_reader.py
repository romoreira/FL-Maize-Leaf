import torchvision.transforms as transforms
import torch
import argparse
import sys
import os
from torch import nn
import torchvision.datasets as datasets
from torchvision import datasets, models, transforms

sys.path.append("../../")
from fedlab.core.client.manager import ActiveClientManager
from fedlab.core.client.trainer import SGDClientTrainer
from fedlab.utils.dataset.sampler import RawPartitionSampler
from fedlab.core.network import DistNetwork
from fedlab.utils.functional import AverageMeter, evaluate

# data_dir = '../dataset/'
#
#
# classes = ['Common_rust', 'Gray_Leaf', 'Healthy', 'Northern_Leaf_Blight']
#
#
#
# train_transforms = transforms.Compose([
#                            transforms.Resize(size=[224, 224]),
#                            transforms.RandomVerticalFlip(0.5),
#                            transforms.RandomRotation(30),
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
#                        ])
#
# test_transforms = transforms.Compose([
#                            transforms.Resize(size=[224, 224]),
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
#                        ])
#
# trains_dir = []
# valids_dir = []
# train_loaders = []
# valid_loaders = []
# test_loaders = []
#
# folds = os.listdir(data_dir + '5-fold')
#
#
# all_size_train = []
# all_size_valid = []
#
# for i in folds:
#     train_dir = os.path.join(data_dir + '5-fold/', i + '/train/')
#     valid_dir = os.path.join(data_dir + '5-fold/', i + '/val/')
#     test_dir = os.path.join(data_dir + '5-fold/', i + '/test/')
#
#
#     train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
#     valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
#     test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
#
#     trainloader = torch.utils.data.DataLoader(train_data, batch_size=32,
#                                                shuffle=True, num_workers=4)
#
#     validloader = torch.utils.data.DataLoader(valid_data, batch_size=32,
#                                                shuffle=True, num_workers=4)
#
#     testloader = torch.utils.data.DataLoader(test_data, batch_size=32,
#                                               shuffle=False, num_workers=4)
#
#     train_loaders.append(trainloader)
#     valid_loaders.append(validloader)
#     test_loaders.append(testloader)
#
#     print("----------------------------------------------------------------------------------------")
#     print(i)
#     print('Num training images: ', len(train_data))
#     print('Num valid images: ', len(valid_data) )
#     print('Num test images: ', len(test_data))
#
#     all_size_train.append(len(train_data))
#     all_size_valid.append(len(valid_data))
#
#     print("----------------------------------------------------------------------------------------")
#     print("\n\n----------------------------------------------------------------------------------------")
#     print("Num train full size:", sum(all_size_train))
#     print("Num valid full size:", sum(all_size_valid))
#     print('Num test images: ', len(test_data), (testloader))
#     print("Num full size (train+valid+test):", sum(all_size_train) + sum(all_size_valid) + len(test_data))

train_transforms = transforms.Compose([
                           transforms.Resize(size=[224, 224]),
                           transforms.RandomVerticalFlip(0.5),
                           transforms.RandomRotation(30),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(size=[224, 224]),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])

data_dir = '../dataset/'
classes = ['Common_rust', 'Gray_Leaf', 'Healthy', 'Northern_Leaf_Blight']
test_dir = os.path.join(data_dir + '5-fold/' + 'test/')
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False, num_workers=args.num_workers)