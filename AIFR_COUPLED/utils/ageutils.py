'''
@Author: Saurav Rai
         IIMtech (CS)
'''
from torch.utils.data import Dataset
from utils import settings
import os
import scipy.io as sio
import pickle
import numpy as np
from sklearn import metrics
from PIL import Image
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.nn.init as init
import math

import time

from utils.agedata import AgeFaceDataset
from utils import settings
	


class CoupledAModel(nn.Module):
        def __init__(self):
                super(CoupledAModel,self).__init__()
                
                #1.THIS IS THE YOUNGER SIDE OF THE ARCHITECTURE: x1 to x1_bar
                '''ENCODER''' 
                self.linear_encod_yng = nn.Linear(32*32,4000)
                #self.sigmoid = nn.F.sigmoid()
                self.sigmoid_yng = nn.Sigmoid()             
                '''DECODER'''
                self.linear_decod_yng = nn.Linear(4000,1024)

                #2.THIS IS THE OLDER SIDE OF THE ARCHITECTURE: x2 to x2_bar
                '''ENCODER''' 
                self.linear_encod_old = nn.Linear(32*32,4000)
                #self.sigmoid = nn.F.sigmoid()
                self.sigmoid_old = nn.Sigmoid()             
                '''DECODER'''
                self.linear_decod_old = nn.Linear(4000,1024)
                self.initWeights()

        def initWeights(self):
	        #This is W1
                init.normal_(self.linear_encod_yng.weight,0,1e-4)
	        #This is b1
                init.constant_(self.linear_encod_yng.bias,0)
	        #This is W1_bar
                init.normal_(self.linear_decod_yng.weight,0,1e-4)
	        #This is c1
                init.constant_(self.linear_decod_yng.bias,0)
	        
                #This is W2 
                init.normal_(self.linear_encod_old.weight,0,1e-4)
	        #This is b2
                init.constant_(self.linear_encod_old.bias,0)
	        #This is W2_bar
                init.normal_(self.linear_decod_old.weight,0,1e-4)
	        #This is c2
                init.constant_(self.linear_decod_old.bias,0)
    		               
        def forward(self,x1,x2):
                #print('shape of image x1',x1.size())               
                x1 = self.linear_encod_yng(x1.view(-1,32*32))
                #print('shape after of image x1',x1.size())               
                x1 = self.sigmoid_yng(x1)
                x1_bar = self.linear_decod_yng(x1)
                
                #print('shape of image x2',x2.size())               
                x2 = self.linear_encod_old(x2.view(-1,32*32))
                #print('shape after of image x2',x2.size())               
                x2 = self.sigmoid_old(x2)
                x2_bar = self.linear_decod_old(x2)
                
                c1 = self.linear_decod_yng.bias
                c2 = self.linear_decod_old.bias
                return  x1_bar , x2_bar , c1 ,c2 
         
def CoupledAutoModel(**kwargs):
    model_auto_encoder = CoupledAModel(**kwargs)
    return model_auto_encoder       

class bridge_model(nn.Module):
        def __init__(self):
                super(bridge_model,self).__init__()
                '''FOR THE AGE PART'''
                #It is used for x1
                #both Wv1 and bv1 will be automatically taken care
                self.linear_encod_x1 = nn.Linear(32*32,800)
                self.sigmoid_x1 = nn.Sigmoid()  

                #after applying this to x1 I get A1(i) 
                #It is used for x2
		#both Wv2 and bv2 will be automatically taken care
                self.linear_encod_x2 = nn.Linear(32*32,800)
                self.sigmoid_x2 = nn.Sigmoid() 
                #after applying this to x2 I get A2(i) 
                     
                '''given the no of neurons is 800'''   
                '''Aging Bridge Network from A1 TO A2 hat''' 
                self.linear_bridge_aging1 = nn.Linear(800,800)
                self.sigmoid_aging1 = nn.Sigmoid()
                
                self.linear_bridge_aging2 = nn.Linear(800,800)
                self.sigmoid_aging2 = nn.Sigmoid()
		
                '''Deaging Bridge Network from A2 TO A1 hat'''
                self.linear_bridge_deaging1 = nn.Linear(800,800)
                self.sigmoid_deaging1 = nn.Sigmoid()
                
                self.linear_bridge_deaging2 = nn.Linear(800,800)
                self.sigmoid_deaging2 = nn.Sigmoid()
    
                '''FOR THE IDENTITY PART'''
                self.linear_identity_yng  = nn.Linear(32*32,2800)
                self.sigmoid_identity_yng = nn.Sigmoid()
                self.linear_identity_old = nn.Linear(32*32,2800)
                self.sigmoid_identity_old = nn.Sigmoid()

                '''THIS IS FOR THE USE OF THE RECONSTRUCTED x1_bar and x2_bar'''
		#this is for the age_young
                self.reconstruted_x1_age = nn.Linear(800,1024)
                #this is for the identity_young
                self.reconstruted_x1_identity = nn.Linear(2800,1024)
		
                #this is for the age_old
                self.reconstruted_x2_age = nn.Linear(800,1024)
                #this is for the identity_old
                self.reconstruted_x2_identity = nn.Linear(2800,1024)
                self.sigmoid_recons = nn.Sigmoid()
               
                self.initWeights()

        
        def initWeights(self):
                #this is Ha1
                init.normal_(self.linear_bridge_aging1.weight,0,1e-4)
                #this is Ha2
                init.normal_(self.linear_bridge_aging2.weight,0,1e-4)
                
                #this is ba1
                init.constant_(self.linear_bridge_aging1.bias,0)
                #this is ba2
                init.constant_(self.linear_bridge_aging2.bias,0)
                #this is Hd1
                init.normal_(self.linear_bridge_deaging1.weight,0,1e-4)
                #this is Hd2
                init.normal_(self.linear_bridge_deaging2.weight,0,1e-4)
                #this is bd1
                init.constant_(self.linear_bridge_deaging1.bias,0)
                #this is bd2
                init.constant_(self.linear_bridge_deaging2.bias,0)
      
        def forward(self,x1,x2,c1,c2):
                '''Age part'''
                #aging part : goes from x1 to a1 to a2_hat
                #print('The size of x1 is',x1.size())
                #print('The size of x2 is',x2.size())
               
                x1_base = x1
                x1 = self.linear_encod_x1(x1)
                a1 = self.sigmoid_x1(x1) #gives me A1(i)
                
                by1 = self.linear_bridge_aging1(a1) #by1 means bridge young 1
                by1 = self.sigmoid_aging1(by1)
                by2 = self.linear_bridge_aging2(by1) #by2 means bridge young 2
                a2_hat = self.sigmoid_aging2(by2)
                #deaging part : goes from x2 to a2 to a1_hat
                x2_base = x2
                x2 = self.linear_encod_x2(x2)
                a2 = self.sigmoid_x2(x2) #gives me A2(i)

                bo1 = self.linear_bridge_aging2(a2) #bo1 means bridge old 1
                bo1 = self.sigmoid_aging1(bo1)
                bo2 = self.linear_bridge_aging2(bo1) #bo2 means bridge old 2
                a1_hat = self.sigmoid_aging2(bo2)
                '''Identity part'''
                id1 = self.linear_identity_yng(x1_base)
                id1 = self.sigmoid_identity_yng(id1)
                id2 = self.linear_identity_yng(x2_base)
                id2 = self.sigmoid_identity_yng(id2)
                
                wu1 = self.linear_identity_yng.weight
                wu2 = self.linear_identity_old.weight
                bu1 = self.linear_identity_yng.bias
                bu2 = self.linear_identity_old.bias
                
                wv1 = self.linear_encod_x1.weight
                wv1 = self.linear_encod_x2.weight
                '''It is for the addition part'''
                
                wv1_hat = self.reconstruted_x1_age.weight
                wu1_hat = self.reconstruted_x1_identity.weight
                wv2_hat = self.reconstruted_x2_age.weight
                wu2_hat = self.reconstruted_x2_identity.weight
                
                mul1_x1 = torch.matmul(wv1_hat,a1_hat.t())
                mul2_x1 = torch.matmul(wu1_hat,id1.t())
                 
                #print('The size of the mul1_x1',mul1_x1.size())
                #print('The size of the mul1_x2',mul2_x1.size())
                #print('The size of the c1',c1.size())
                
                mul1_x1 = mul1_x1.t()
                mul2_x1 = mul2_x1.t()

                add_mul1 = torch.add(mul1_x1,1,mul2_x1)
                add_x1 = torch.add(add_mul1,1,c1) 

                reconstruct_x1 = self.sigmoid_recons(add_x1)     
                #print('reconstructed x1',reconstruct_x1.size())
                mul1_x2 = torch.matmul(wv2_hat,a2_hat.t())
                mul2_x2 = torch.matmul(wu2_hat,id2.t())
                
                mul1_x2 = mul1_x2.t()
                mul2_x2 = mul2_x2.t()
                
                add_mul2 = torch.add(mul1_x2,1,mul2_x2)
                add_x2 = torch.add(add_mul2,1,c2) 
                
                #print('The size of the mul2_x1',mul1_x2.size())
                #print('The size of the mul2_x2',mul2_x2.size())
                #print('The size of the c2',c2.size())
                
                reconstruct_x2 = self.sigmoid_recons(add_x2)     
                #print('reconstructed x2',reconstruct_x2.size())
                
                
                return a1 , a1_hat , a2 ,a2_hat , id1 ,id2 , x1_base , reconstruct_x1 ,x2_base , reconstruct_x2  , wu1 , wu2 #, wv1 , wv2, bu1 ,bu2
                #note x1_base and x2_base are original x1 and x2 values
def Bridge_Model(**kwargs):
    bridgemodel = bridge_model(**kwargs)
    return bridgemodel       
                
class Test_model_yng(nn.Module):
    def __init__(self):
        super(Test_model_yng,self).__init__()
        self.linear_encod_yng = nn.Linear(1024,2800)
        self.sigmoid = nn.Sigmoid()
        
        pre_bridge_dict = torch.load('./1bridgemodel45_checkpoint.pth.tar',map_location = lambda storage, loc: storage)
        self.pre_trained_dict = pre_bridge_dict['state_dict']
        self.initWeights()
    
    def initWeights(self):
       
        self.linear_encod_yng.weight = nn.Parameter(self.pre_trained_dict['module.linear_identity_yng.weight'])
        self.linear_encod_yng.bias =  nn.Parameter(self.pre_trained_dict['module.linear_identity_yng.bias'])
        
        '''
        self.linear_encod_yng.weight = self.pre_trained_dict['module.linear_identity_yng.weight']
        self.linear_encod_yng.bias =  self.pre_trained_dict['module.linear_identity_yng.bias']
        '''
    def forward(self,x1):
                x1 = self.linear_encod_yng(x1)
                x1 = self.sigmoid(x1)
                
                features_yng_wgt = self.linear_encod_yng.weight
                features_yng_bias = self.linear_encod_yng.bias
             
                return x1 , features_yng_wgt ,features_yng_bias  #feature vector of young image
                
class Test_model_old(nn.Module):
    def __init__(self):
        super(Test_model_old,self).__init__()
        self.linear_encod_old = nn.Linear(1024,2800)
        self.sigmoid = nn.Sigmoid()
        
        pre_bridge_dict = torch.load('./1bridgemodel45_checkpoint.pth.tar',map_location = lambda storage, loc: storage) #after generating the bridge model need to use for the test model.
        self.pre_trained_dict = pre_bridge_dict['state_dict']
        self.initWeights()
    
    def initWeights(self):
     
        self.linear_encod_old.weight = nn.Parameter(self.pre_trained_dict['module.linear_identity_old.weight'])
        self.linear_encod_old.bias =  nn.Parameter(self.pre_trained_dict['module.linear_identity_old.bias'])
    
    def forward(self,x2):
                x2 = self.linear_encod_old(x2)
                x2 = self.sigmoid(x2)
                
                features_old_wgt = self.linear_encod_old.weight
                features_old_bias = self.linear_encod_old.bias
                 
                return x2 ,features_old_wgt , features_old_bias  #feature vector of old image


def Test_Model_yng(**kwargs):
    testmodel_yng = Test_model_yng(**kwargs)
    return testmodel_yng       


def Test_Model_old(**kwargs):
    testmodel_old = Test_model_yng(**kwargs)
    return testmodel_old       


def save_checkpoint(state,  filename):
    torch.save(state, filename)


def train_basic_step(train_loader, coupledautomodel, criterion, optimizer, epoch, device):
    
    running_loss = 0.   
    
    data_size = 0
    
    #lightcnnmodel.train(True)
    
    for (label,x1,x2) in train_loader:
        
        optimizer.zero_grad()
        
        x1 = x1.to(device)
        x2 = x2.to(device)
                
        x1 = x1.view(-1,32*32)
        x2 = x2.view(-1,32*32)

            
        label = torch.tensor(label, dtype = torch.long)
        label = label.to(device)
       
        
        recons_x1, recons_x2 , __ , __  = coupledautomodel(x1,x2)   #added by dg on 27-08 

        #LOSS FUNCTION IN THE AUTOENCODER 
        loss_x1  = criterion(x1,recons_x1)
        loss_x2  = criterion(x2,recons_x2)
        loss = loss_x1 +  loss_x2   

        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item() * label.size(0)
        data_size += label.size(0)
    #print('Basic training step loss in the ageutils.py:',running_loss/data_size)
    return running_loss / data_size

def train_transfer_step(train_loader,coupledautomodel,bridgemodel, criterion, optimizer, epoch, device):
    
    running_loss = 0.   
    
    data_size = 0
    
    #lightcnnmodel.train(True)
    
    for (label,x1,x2) in train_loader:
        
        optimizer.zero_grad()
        
        x1 = x1.to(device)
        x2 = x2.to(device)
         
        x1 = x1.view(-1,32*32)
        x2 = x2.view(-1,32*32)
            
        label = torch.tensor(label, dtype = torch.long)
        label = label.to(device)
       
        
        __, __ , c1 , c2  = coupledautomodel(x1,x2)   #added by dg on 27-08 
        
        a1 , a1_hat , a2, a2_hat ,id1 ,id2 , x1, recons_x1 , x2,recons_x2,wu1,wu2 = bridgemodel(x1,x2,c1,c2)
        #LOSS FUNCTION IN THE BRIDGE NETWORKS
        
        loss_a1 = criterion(a1,a1_hat)       
        loss_a2 = criterion(a2,a2_hat)    
        loss_id = criterion(id1,id2)        
        loss_x1 = criterion(x1,recons_x1)
        loss_x2 = criterion(x2,recons_x2)
        
        total_loss = loss_a1 + loss_a2 + loss_id + loss_x1 + loss_x2 
        
        total_loss.backward()
        
        optimizer.step()
        
        running_loss += total_loss.item() * label.size(0)
        data_size += label.size(0)
    #print('Transfer step loss in the ageutils.py file: ',running_loss/data_size)
    return running_loss / data_size

    
def mytest_gall(test_query_loader , test_gall_loader ,coupledautomodel,bridgemodel,testmodel_yng,testmodel_old,device):
    
    acc = 0
    target =[]
    target2 =[]
    query_features = []
    gallery_features = []
    q_features = []
    g_features = []
    features_yng_wt = []
    features_yng_bias = []
    wt = []
    bias = []   
    print('MOST RECENT')
    with torch.no_grad():
       
        #CHAGED BY SAURAV 3/10/19
        for (x1,age1,label1) in test_query_loader:
            x1 = x1.to(device)
            x1 = x1.view(-1,32*32)
            
            x1_recons_feat,wt_features,bias_features  = testmodel_yng(x1)
            label1 = torch.tensor(label1, dtype = torch.long)
            label1 = label1.to(device)
            #print('The len of reconstructed x1',len(x1_recons_feat))
            for j in range(len(x1_recons_feat)):
                q_features.append(x1_recons_feat[j].cpu().numpy())
                target.append(label1[j].cpu().numpy())

            for j in range(len(wt_features)):
                features_yng_wt.append(wt_features[j].cpu().numpy())   
 
            for j in range(len(bias_features)):
                features_yng_bias.append(bias_features[j].cpu().numpy())    
    
        for (x2,age2,label2) in test_gall_loader:
            x2 = x2.to(device)
            x2 = x2.view(-1,32*32)

            x2_recons_feat, _ , _  = testmodel_old(x2)
            label2 = torch.tensor(label2, dtype = torch.long)
            label2 = label2.to(device)
            #print('The len of reconstructed x2',len(x2_recons_feat))
            
            for j in range(len(x2_recons_feat)):
               
                g_features.append(x2_recons_feat[j].cpu().numpy())
                target2.append(label2[j].cpu().numpy()) 
       
        total = len(q_features)
        
        q_features = np.array(q_features)
        g_features = np.array(g_features)
                
        wt = np.array(features_yng_wt)
        bias = np.array(features_yng_bias)
        print('The identity features wt yng',wt)    
        print('The identity features bias yng',bias)    

        #print('The query features ',q_features)
        #print('The gallery features ',g_features)
        print('The query features size',q_features.shape)
        print('The gallery features size',g_features.shape)
        
        target = np.array(target)
        target2 = np.array(target2)
        
        dist  = metrics.pairwise.cosine_similarity(q_features, g_features)
        print('THe dist shape',dist.shape)
        Avg_Prec =0
        Total_average_precision = 0
        
        #WE WILL BE SELECTING THE 5 CLOSEST FEATURES 
        k=5

        for i in range(len(q_features)):
            correct_count = 0
            prec = 0
            
            idx = np.argpartition(dist[i],-k)
            indices = idx[-k:] #THIS WILL GIVE ME THE INDICES OF 5 HIGHEST VALUE         
            #print('THe indices',indices)
            true_label  = target[i]
            #print('The true label',true_label)
            
          
            for j in range(1,len(indices)+1):
                #print('The target2 values are:',target2[indices[j-1]])
                if(true_label == target2[indices[j-1]]):
                    correct_count = correct_count + 1
                    prec = prec + float(correct_count) / j
                if(correct_count == 0):
                    correct_count =1
                    prec = 0
            #print('The value of correct_count',correct_count)
            #print('The value of precision',prec)
            Avg_Prec = Avg_Prec + 1.0/correct_count * prec
       
        Total_average_precision =  Avg_Prec
       
        Mean_Average_Precision = 1/total * Total_average_precision
        
    return Mean_Average_Precision    

'''
        output = np.argmax(dist,axis=1)
        
        
        correct =np.sum(target2[output] ==target) 
        print('correct and total',correct,total)
        acc = correct * 100.0/total
        
    return  acc
'''


def adjust_learning_rate(optimizer, epoch):
    
    for param_group in optimizer.param_groups:
        if epoch > 3:
            param_group['lr'] = 0.0001
                
            
            
            
            
    
