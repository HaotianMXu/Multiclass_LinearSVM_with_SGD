#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear SVM with custom multiclass Hinge loss
"""

import os
import time
import argparse
import pickle
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,gradcheck
import torch.utils.data as utils

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle
"""training settings"""
parser = argparse.ArgumentParser(description='Linear SVM with SGD')
parser.add_argument('--C',type=float, default=1.0,metavar='C',
                    help='penalty parameter C of the error term')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=60, metavar='N',
                    help='input batch size for testing (default: 10000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=666, metavar='S',
                    help='random seed (default: 666)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
"""build datasets"""
def syntheticData():#generate fake data
    #class 0
    X_train0_0=np.random.uniform(low=0.0,high=1.0,size=200)
    X_train0_1=np.random.uniform(low=2.0,high=3.0,size=200)
    X_train0=np.concatenate((X_train0_0.reshape(-1,1),X_train0_1.reshape(-1,1)),axis=1)
    y_train0=np.zeros(200,dtype=np.uint8)
    #class 1
    X_train1_0=np.random.uniform(low=0.0,high=1.0,size=200)
    X_train1_1=np.random.uniform(low=2.9,high=3.9,size=200)
    X_train1=np.concatenate((X_train1_0.reshape(-1,1),X_train1_1.reshape(-1,1)),axis=1)
    y_train1=np.ones(200,dtype=np.uint8)
    #class 2
    X_train2_0=np.random.uniform(low=0.9,high=1.9,size=200)
    X_train2_1=np.random.uniform(low=2.0,high=3.0,size=200)
    X_train2=np.concatenate((X_train2_0.reshape(-1,1),X_train2_1.reshape(-1,1)),axis=1)
    y_train2=2*np.ones(200,dtype=np.uint8)
    #concat
    X_train=np.concatenate((X_train0[:180],X_train1[:180],X_train2[:180]),axis=0).astype(np.float32)
    y_train=np.concatenate((y_train0[:180],y_train1[:180],y_train2[:180]),axis=0)
    X_train,y_train=shuffle(X_train,y_train)
    X_test=np.concatenate((X_train0[180:],X_train1[180:],X_train2[180:]),axis=0).astype(np.float32)
    y_test=np.concatenate((y_train0[180:],y_train1[180:],y_train2[180:]),axis=0)
    
    n_feature=2
    n_class=3
    y_train_list=y_train.tolist()
    y_test_list=y_test.tolist()
    
    #loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    trn_loader=utils.DataLoader(utils.TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train)),batch_size=args.batch_size,shuffle=True,**kwargs)
    tst_loader=utils.DataLoader(utils.TensorDataset(torch.from_numpy(X_test),torch.from_numpy(y_test)),batch_size=args.test_batch_size,shuffle=False,**kwargs)
    
    return trn_loader,tst_loader,n_feature,n_class,y_train_list,y_test_list

def buildDataLoader(path):#load real data
    with open(path,'rb') as f:
        data=pickle.load(f)
    
    X_train=torch.from_numpy(data['X_train'])
    y_train=torch.from_numpy(data['y_train'])
    X_test=torch.from_numpy(data['X_test'])
    y_test=torch.from_numpy(data['y_test'])
    
    y_train_list=data['y_train'].tolist()
    y_test_list=data['y_test'].tolist()
    
    n_feature=data['X_train'].shape[1]
    n_class=np.asscalar(np.amax(data['y_train'])+1)#numpy.int64 to python int
    
    del data
    gc.collect()
    print("data loaded!")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    trn_set=utils.TensorDataset(X_train,y_train)
    trn_loader=utils.DataLoader(trn_set,batch_size=args.batch_size,shuffle=True,**kwargs)
    
    tst_set=utils.TensorDataset(X_test,y_test)
    tst_loader=utils.DataLoader(tst_set,batch_size=args.test_batch_size,shuffle=False,**kwargs)   
    
    return trn_loader,tst_loader,n_feature,n_class,y_train_list,y_test_list

"""define network"""
class Net(nn.Module):
    def __init__(self,n_feature,n_class):
        super(Net, self).__init__()
        self.fc=nn.Linear(n_feature,n_class)
        torch.nn.init.kaiming_uniform(self.fc.weight)
        torch.nn.init.constant(self.fc.bias,0.1)
        
    def forward(self,x):
        output=self.fc(x)
        return output
"""SVM loss
Weston and Watkins version multiclass hinge loss @ https://en.wikipedia.org/wiki/Hinge_loss
for each sample, given output (a vector of n_class values) and label y (an int \in [0,n_class-1])
loss = sum_i(max(0, (margin - output[y] + output[i]))^p) where i=0 to n_class-1 and i!=y

Note: hinge loss is not differentiable
      Let's denote hinge loss as h(x)=max(0,1-x). h'(x) does not exist when x=1, 
      because the left and right limits do not converge to the same number, i.e.,
      h'(1-delta)=-1 but h'(1+delta)=0.
      
      To overcome this obstacle, people proposed squared hinge loss h2(x)=max(0,1-x)^2. In this case,
      h2'(1-delta)=h2'(1+delta)=0
"""
class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average
    def forward(self, output, y):#output: batchsize*n_class
        #print(output.requires_grad)
        #print(y.requires_grad)
        output_y=output[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()].view(-1,1)#view for transpose
        #margin - output[y] + output[i]
        loss=output-output_y+self.margin#contains i=y
        #remove i=y items
        loss[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()]=0
        #max(0,_)
        loss[loss<0]=0
        #^p
        if(self.p!=1):
            loss=torch.pow(loss,self.p)
        #add weight
        if(self.weight is not None):
            loss=loss*self.weight
        #sum up
        loss=torch.sum(loss)
        if(self.size_average):
            loss/=output.size()[0]#output.size()[0]
        return loss
"""gradcheck"""
"""
Check gradients computed via small finite differences against analytical gradients
The check between numerical and analytical has the same behaviour as numpy.allclose 
https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
meaning it check that
        absolute(a - n) <= (atol + rtol * absolute(n))
is true for all elements of analytical jacobian a and numerical jacobian n.

    Args:
        func: Python function that takes Variable inputs and returns
            a tuple of Variables
        inputs: tuple of Variables
        eps: perturbation for finite differences
        atol: absolute tolerance
        rtol: relative tolerance
    Returns:
        True if all differences satisfy allclose condition
"""
def gradientCheck():
    fake_lbl=np.random.randint(10, size=100)
    fake_output=np.random.normal(loc=0.0, scale=1.0, size=10*100).reshape((100,10))
    inp=(Variable(torch.from_numpy(fake_output).double().cuda(),requires_grad=True),Variable(torch.from_numpy(fake_lbl).long().cuda(),requires_grad=False))
    tt=gradcheck(multiClassHingeLoss(),inp)
    if(tt):
        print('gradcheck for loss function---pass')
    else:
        print('gradcheck for loss function---FAIL')

    fake_input=np.random.normal(loc=0.0, scale=1.0, size=2*100).reshape((100,2))
    inp=(Variable(torch.from_numpy(fake_input).float().cuda(),requires_grad=True),)#Variable(torch.from_numpy(fake_lbl).long().cuda(),requires_grad=False))
    tt_m=gradcheck(model,inp,eps=1e-3)#1e-3, since we use float instead of double here
    if(tt_m):
        print('gradcheck for model---pass')
    else:
        print('gradcheck for model---FAIL')
    
    time.sleep(5)
	return(tt and tt_m)
"""training"""
def train(epoch):
    model.train()
    training_loss=0
    training_f1=0
    for batch_idx,(data,target) in enumerate(trn_loader):
        if args.cuda:
            data,target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)
        optimizer.zero_grad()
        output=model(data)
        #output1=output#print(output)
        #target1=target#print(target)
        #tloss=weighted_binary_cross_entropy(output,target,wei)
        tloss=loss(output,target)##focalLoss(output,target)#focalLoss(output,target)#F.nll_loss(output,target)#
        training_loss+=tloss.data[0]
        tloss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        training_f1+=f1_score(target.data.cpu().numpy(),pred.cpu().numpy(),labels=np.arange(n_class).tolist(),average='macro')
        #if batch_idx % args.log_interval==0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(trn_loader.dataset),
        #        100. * batch_idx / len(trn_loader), tloss.data[0]))
    print('Epoch: {}'.format(epoch))
    print('Training set avg loss: {:.4f}'.format(training_loss/len(trn_loader)))
    print('Training set avg micro-f1: {:.4f}'.format(training_f1/len(trn_loader)))
"""testing"""
def test():
    model.eval()
    test_loss = 0
    preds=list()
    for data, target in tst_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        preds+=pred.cpu().numpy().tolist()
    test_loss /= len(tst_loader.dataset)
    conf_mat=confusion_matrix(y_test_list,preds)
    precision,recall,f1,sup=precision_recall_fscore_support(y_test_list,preds,average='macro')
    print('Test set avg loss: {:.4f}'.format(test_loss))
    print('conf_mat:\n',conf_mat)
    print('Precison:{:.4f}\nRecall:{:.4f}\nf1:{:.4f}\n'.format(precision,recall,f1))    
    return conf_mat,precision, recall, f1

def main():
"""init model"""
    trn_loader,tst_loader,n_feature,n_class,y_train_list,y_test_list=syntheticData()
    model=Net(n_feature,n_class)
    #model.apply(weights_init)
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss=multiClassHingeLoss()
    testloss=multiClassHingeLoss()

"""begin to train"""
    best_epoch_idx=-1
    best_f1=0.
    history=list()
    for epoch in range(0,args.epochs):
        train(epoch)
        conf_mat, precision, recall, f1=test() 
        history.append((conf_mat, precision, recall, f1))
        if f1>best_f1:#save best model
            best_f1=f1
            best_epoch_idx=epoch
            torch.save(model.state_dict(),'best.model')

    print('Best epoch:{}\n'.format(best_epoch_idx))
    conf_mat, precision, recall, f1=history[best_epoch_idx]
    print('conf_mat:\n',conf_mat)
    print('Precison:{:.4f}\nRecall:{:.4f}\nf1:{:.4f}\n'.format(precision,recall,f1))    

if __name__="__main__":
    main()
    
    
    
    
