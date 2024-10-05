import sys
sys.path.append("..")

from utils.Dataset import *
from model.ResNet import *
from torch.utils.data import DataLoader
from random import shuffle
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

class Client():
    '''A Client in FL System'''
    '''Initialize the member of Client'''
    def __init__(self,uid,model,dataset_name,train_idx,test_idx):
        self.uid=uid
        self.model=model
        self.dataset_name=dataset_name
        self.train_idx=list(train_idx)
        shuffle(self.train_idx) # necessary!!!
        self.test_idx=list(test_idx)
    '''Local Training 10 epoch'''
    def ClientLocalTraining(self,device,train_dataset,train_idx):
        # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        loss_func=nn.CrossEntropyLoss()

        # loss_func = nn.NLLLoss().to(device)
        # self.criterion_cosin = CosineSimilarityLoss(args.lambda1)

        optimizer=torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # optimizer=torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        if self.dataset_name=="Speech-Commands":
            train_loader=DataLoader(DatasetSplitSpeech(train_dataset,train_idx),batch_size=64,shuffle=True)
        else:
            train_loader=DataLoader(DatasetSplit(train_dataset,train_idx),batch_size=64,shuffle=True)
        for epoch in range(10):
            print("[client uid:%d,epoch:%d] Local Training..."%(self.uid,epoch))
            for iter_,(xb,yb) in enumerate(train_loader,0):
                xb,yb=xb.to(device),yb.to(device)
                optimizer.zero_grad()
                pred=self.model(xb)
                loss=loss_func(pred,yb)
                loss.backward()
                optimizer.step()
        return self.model.state_dict()
    '''Test for Accuracy'''
    def ClientLocalTesting(self,device,test_dataset,test_idx):
        self.model.to(device)
        self.model.eval()
        total,correct=0.0,0.0
        if self.dataset_name=="Speech-Commands":
            test_loader=DataLoader(DatasetSplitSpeech(test_dataset,test_idx),batch_size=64,shuffle=True)
        else:
            test_loader=DataLoader(DatasetSplit(test_dataset,test_idx),batch_size=64,shuffle=True)
        for iter_,(xb,yb) in enumerate(test_loader,0):
            xb,yb=xb.to(device),yb.to(device)
            pred=self.model(xb).argmax(dim=1)
            total+=xb.size(0)
            correct+=torch.eq(pred,yb).sum().item()
        acc=correct*1.0/total
        return acc

def get_ClientSet(num_clients,family_name,dataset_name,train_group,test_group):
    client_dict={}
    num_classes=0
    if dataset_name=="CIFAR-10" or dataset_name=="CINIC-10":
        input_channel=3
        w,h=32,32
        num_classes=10
    
    if family_name=="ResNet":
        for uid in range(num_clients):
            if uid % 3 == 0:
                client_dict[uid]=Client(uid,resnet50(num_classes=num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
            elif uid % 3 == 1:
                client_dict[uid]=Client(uid,resnet101(num_classes=num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
            elif uid % 3 == 2:
                client_dict[uid]=Client(uid,resnet152(num_classes=num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
        return client_dict

# num_clients=8
# train_dataset,test_dataset,train_group,test_group=get_dataset("CIFAR-10",num_clients)
# dict=get_ClientSet(num_clients,"VGG",train_group,test_group)
# print(dict[5].model)
