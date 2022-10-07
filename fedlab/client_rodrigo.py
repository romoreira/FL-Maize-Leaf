from time import time
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

import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns


model_name = "vgg"
num_classes = 4 
feature_extract = True

GPUavailable = torch.cuda.is_available()
if GPUavailable:
   print('Treinamento em GPU!')
   device = torch.device("cuda:0")
else:
   print('Treinamento em CPU!')
   device = "cpu"

"""
# torch model
class MLP(nn.Module):

    def __init__(self, input_size=150528, output_size=32):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""
class AsyncClientTrainer(SGDClientTrainer):

    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 cuda=True,
                 logger=None):
        super().__init__(model, data_loader, epochs, optimizer, criterion,
                         cuda, logger)
        self.time = 0

    def local_process(self, payload):
        self.time = payload[1].item()
        return super().local_process(payload)

    @property
    def uplink_package(self):
        return [self.model_parameters, torch.Tensor([self.time])]


def matriz_confusao(correct, pred, nome, x):
      correct_list = []
      predict_list = []

      for i in correct:
          correct_list.extend(i)

      for j in pred:
          predict_list.extend(j)


      print("Listas: ")
      print("Correct: ", correct_list)
      print("Predict: ", predict_list)
      print("\n")

      print("Matriz de Confusão ("+nome+") do "+x+": ")
      cm = confusion_matrix(correct_list, predict_list)
      print(cm)

      ax = plt.subplot()
      sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d") #annot=True to annotate cells
      ax.set_title('Matriz de Confusão ('+nome+') ('+x+')')
      plt.savefig(resultados_dir+'/'+'matrizconfusao_'+nome[:3]+'_'+x+'.pdf')
      plt.close()
      plt.show()

      print("\nRelatório de classificação ("+nome+"): ")
      report = metrics.classification_report(correct_list, predict_list, target_names=classes, digits=4)
      print(metrics.classification_report(correct_list, predict_list, target_names=classes))

      file_report = open('./Resultados/'+model_name+'/report_matrix.txt', 'a+')
      file_report.write("%s \n" %nome)
      file_report.write("%s \n" %x)
      file_report.write(report)
      file_report.write("\n")
      file_report.write("Matriz de Confusão \n")
      file_report.write(str(cm))
      file_report.write("\n \n")
      file_report.close()

      '''
       # calculate the fpr and tpr for all thresholds of the classification
      fpr, tpr, threshold = metrics.roc_curve(correct_list, predict_list)
      roc_auc = metrics.auc(fpr, tpr)

      # method I: plt
      plt.title('Receiver Operating Characteristic')
      plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
      plt.legend(loc = 'lower right')
      plt.plot([0, 1], [0, 1],'r--')
      plt.xlim([0, 1])
      plt.ylim([0, 1])
      plt.ylabel('True Positive Rate')
      plt.xlabel('False Positive Rate')
      plt.savefig(resultados_dir+'/'+'ROC_'+nome[:3]+'_'+x+'.pdf')
      plt.close()
      #plt.show()
      '''


# Listas --------------------------
train_correct_list = []
train_predict_list = []

valid_correct_list = []
valid_predict_list = []

test_correct_list = []
test_predict_list = []
#----------------------------------


name = './Resultados'
if os.path.isdir(name) == False:
        os.mkdir(name)

resultados_dir = './Resultados/'+model_name
if os.path.isdir(resultados_dir) == False:
        os.mkdir(resultados_dir)

def conf_matrix(fx, y, nome):

        if(nome == 'treino'):

          preds = fx.max(1, keepdim=True)[1]
          correct = y

          c = correct.tolist()
          p = preds.flatten().tolist()

          train_correct_list.append(c)
          train_predict_list.append(p)

          return train_correct_list, train_predict_list

        if(nome == 'validacao'):

          preds = fx.max(1, keepdim=True)[1]
          correct = y

          c = correct.tolist()
          p = preds.flatten().tolist()

          valid_correct_list.append(c)
          valid_predict_list.append(p)

          return valid_correct_list, valid_predict_list

        if(nome == 'teste'):

          preds = fx.max(1, keepdim=True)[1]
          correct = y

          c = correct.tolist()
          p = preds.flatten().tolist()

          test_correct_list.append(c)
          test_predict_list.append(p)

          return test_correct_list, test_predict_list

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = True

parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)

parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--num_class", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=4)

args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

data_dir = '../dataset/'
classes = ['Common_rust', 'Gray_Leaf', 'Healthy', 'Northern_Leaf_Blight']

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

trains_dir = []
valids_dir = []
train_loaders = []
valid_loaders = []
test_loaders = []

folds = os.listdir(data_dir + '5-fold')


all_size_train = []
all_size_valid = []

for i in folds:
    train_dir = os.path.join(data_dir + '5-fold/', i + '/train/')
    valid_dir = os.path.join(data_dir + '5-fold/', i + '/val/')
    test_dir = os.path.join(data_dir + '5-fold/', i + '/test/')


    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4)

    validloader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4)

    train_loaders.append(trainloader)
    valid_loaders.append(validloader)
    test_loaders.append(testloader)

    print("----------------------------------------------------------------------------------------")
    print(i)
    print('Num training images: ', len(train_data))
    print('Num valid images: ', len(valid_data) )
    print('Num test images: ', len(test_data))

    all_size_train.append(len(train_data))
    all_size_valid.append(len(valid_data))

    print("----------------------------------------------------------------------------------------")
    print("\n\n----------------------------------------------------------------------------------------")
    print("Num train full size:", sum(all_size_train))
    print("Num valid full size:", sum(all_size_valid))
    print('Num test images: ', len(test_data), (testloader))
    print("Num full size (train+valid+test):", sum(all_size_train) + sum(all_size_valid) + len(test_data))






def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        # Inicializando cada variável específica para cada modelo
        model_ft = None
        input_size = 0
        print("PRocurando model_name: "+str(model_name))
        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "resnext101_32x8d":
            """ resnext101_32x8d
            """
            model_ft = torch.hub.load('pytorch/vision:v0.9.0', 'resnext101_32x8d', pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "densenet":
            """ Densenet """
            model_ft = models.densenet121(pretrained=use_pretrained)
            #model_ft = models.densenet201(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "mobilenet":
            model_ft = models.mobilenet_v2(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "shufflenet":
            model_ft = models.shufflenet_v2_x1_0(pretrained=True)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained, aux_logits = False)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299
        
        elif model_name == "mlp":
             model_ft = MLP()
             return model_ft

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

optimizer = torch.optim.SGD(model_ft.parameters(), lr=args.lr)
#optimizer = torch.optim.Adam(model_ft.parameters(), lr=args.lr)

criterion = nn.CrossEntropyLoss()
handler = AsyncClientTrainer(model_ft,
                             trainloader,
                             epochs=args.epoch,
                             optimizer=optimizer,
                             criterion=criterion,
                             cuda=args.cuda)

network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank)

t = time()


pytorch_total_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
print("Number of trainable parameters: "+str(pytorch_total_params))

Manager = ActiveClientManager(trainer=handler, network=network)
Manager.run()

tend = time()

print("Final Score Client: "+str(evaluate(model_ft, criterion, testloader)))
print((tend - t)/60)

print(len(test_loaders))
iterator = test_loaders[0]

c = ""
p = ""
with torch.no_grad():
   for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)
        fx = model_ft(x)
        c, p = conf_matrix(fx, y, "validacao")


# Matriz de Confusão para teste
print("===========================================================================================")
matriz_confusao(c, p, 'teste', "Round_1")
print("===========================================================================================")
