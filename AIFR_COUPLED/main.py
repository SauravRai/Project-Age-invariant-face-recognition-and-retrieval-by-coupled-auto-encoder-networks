"""
@author: Saurav Rai

Pytorch Implementation of the paper Age invariant face recognition and retrieval by coupled auto-encoder networks NeuroComputing, 2017, Chenfei Xu
"""
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms

import time

from utils.agedata import AgeFaceDataset
from torch.utils.data import Dataset, DataLoader
from utils import settings

from utils.ageutils import CoupledAutoModel , Bridge_Model,Test_Model_yng,Test_Model_old,adjust_learning_rate
from utils import ageutils
        
def main():
    
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    couplemodel = CoupledAutoModel()
    couplemodel = nn.DataParallel(couplemodel).to(device)
    
    bridgemodel = Bridge_Model()
    bridgemodel = nn.DataParallel(bridgemodel).to(device)
    
    testmodel_yng = Test_Model_yng()
    testmodel_yng = nn.DataParallel(testmodel_yng).to(device)
     
    testmodel_old = Test_Model_old()
    testmodel_old = nn.DataParallel(testmodel_old).to(device)
 
    params = []
   
    for name, param in couplemodel.named_parameters():
    	params.append(param) 
           
    for name, param in bridgemodel.named_parameters():
    	params.append(param)            
    
    for name, param in testmodel_yng.named_parameters():
    	params.append(param)            
    
    for name, param in testmodel_old.named_parameters():
    	params.append(param)            
    	            
        
    
    optimizer = torch.optim.SGD(params, lr = settings.args.lr, momentum = 0.9)
    
    train_transform = transforms.Compose([transforms.Resize(35,interpolation=4), transforms.RandomCrop(32), transforms.ToTensor()])     
    #train_transform = transforms.Compose([transforms.Resize(35), transforms.RandomCrop(32), transforms.ToTensor()])     
    
    valid_transform = transforms.Compose([transforms.Resize(32,interpolation=4), transforms.ToTensor()])    
    
    
    #TRAIN LOADER
    train_loader = DataLoader(AgeFaceDataset(transform = train_transform, istrain = True, isvalid = False , isquery = False ,
                    isgall1 = False ,isgall2 = False ,isgall3 =False),
                    batch_size = settings.args.batch_size, shuffle = True,
                    num_workers = settings.args.workers, pin_memory = False)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode ='min',factor =0.1, patience= 5)
   
    '''
    #VALID LOADER
    
    valid_query_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = True, isquery = True,
                    isgall1 = False , isgall2 = False ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 
    
    valid_gall1_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = True, isquery = False,
                    isgall1 = True , isgall2 = False ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 
  
    valid_gall2_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = True, isquery = False,
                    isgall1 = False , isgall2 = True ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 
   
    valid_gall3_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = True, isquery = False,
                    isgall1 = False , isgall2 = False ,isgall3 = True), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 
    '''
    #TEST LOADER
    test_query_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = False, isquery = True,
                    isgall1 = False , isgall2 = False ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 

    test_gall1_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = False, isquery = False,
                    isgall1 = True , isgall2 = False ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False)
    
    test_gall2_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = False, isquery = False,
                    isgall1 = False , isgall2 = True ,isgall3 = False), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False)
     
    test_gall3_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = False ,isquery = False,
                    isgall1 = False , isgall2 = False ,isgall3 = True), 
                    batch_size = settings.args.batch_size, shuffle = False,
                    num_workers = settings.args.workers, pin_memory = False) 

    
    criterion = nn.MSELoss().to(device)
    
    for epoch in range(settings.args.start_epoch, settings.args.epochs):
    
        adjust_learning_rate(optimizer, epoch)
        for param_group in optimizer.param_groups:
            print('lr after', param_group['lr'])	        

        
        test_mean_avg_prec1 = ageutils.mytest_gall(test_query_loader,test_gall1_loader,couplemodel ,bridgemodel,testmodel_yng,testmodel_old,device)
        print('test mean average precision for gallery1 : {:8f}'.format(test_mean_avg_prec1))
        '''
        test_mean_avg_prec2 = ageutils.mytest_gall(test_query_loader,test_gall2_loader, couplemodel, bridgemodel,testmodel_yng,testmodel_old ,device)
        print('test mean average precision for gallery2 : {:8f}'.format(test_mean_avg_prec2))
        test_mean_avg_prec3 = ageutils.mytest_gall(test_query_loader,test_gall3_loader, couplemodel, bridgemodel,testmodel_yng,testmodel_old,device)
        print('test mean average precision for gallery3 : {:8f}'.format(test_mean_avg_prec3))
        ''' 
        '''
        start = time.time()
        epoch_loss1 = ageutils.train_basic_step(train_loader, couplemodel, criterion, optimizer, epoch, device)

        print('\n train_basic_loss: {:.6f}, Epoch: {:d} \n'.format(epoch_loss1,epoch))

        epoch_loss2 = ageutils.train_transfer_step(train_loader, couplemodel, bridgemodel, criterion, optimizer, epoch, device)
        print('\n transfer_step_loss: {:.6f}, Epoch: {:d} \n'.format(epoch_loss2, epoch ))

        epoch_loss = epoch_loss1 + epoch_loss2

        end = time.time()

        print('\n total_loss: {:.6f}, Epoch: {:d} Epochtime:{:2.2f}\n'.format(epoch_loss, epoch + 1, (end - start)))
        '''

        '''
        valid_mean_avg_prec1 = ageutils.mytest_gall(valid_query_loader,valid_gall1_loader, couplemodel, bridgemodel, device)
        print('test mean average precision for gallery1(valid) : {:8f}'.format(valid_mean_avg_prec1))
        valid_mean_avg_prec2 = ageutils.mytest_gall(valid_query_loader,valid_gall2_loader, couplemodel, bridgemodel, device)
        print('test mean average precision for gallery2(valid) : {:8f}'.format(valid_mean_avg_prec2))
        valid_mean_avg_prec3 = ageutils.mytest_gall(valid_query_loader,valid_gall3_loader, couplemodel, bridgemodel, device)
        print('test mean average precision for gallery3(valid) : {:8f}'.format(valid_mean_avg_prec3))
        
        scheduler.step(epoch_loss)        
        '''        
        
        if epoch % 45 == 0 :    
            
            save_name = settings.args.save_path + 'bridgemodel' + str(epoch) + '_checkpoint.pth.tar'
            
            #ageutils.save_checkpoint({'epoch': epoch + 1,
    	                         #'state_dict': dresmodel.state_dict()}, '1'+ save_name)
            #ageutils.save_checkpoint({'epoch': epoch ,
                             #'state_dict': lightcnnmodel.state_dict()}, '3'+ save_name) # commented by bala on 22-7-18
            # following line added by bala on 22-07-2018 
            
            #ageutils.save_checkpoint({'epoch': epoch,
            #                 'state_dict': lightcnnmodel.state_dict(), 'optimizer': optimizer.state_dict()}, 
            #                 '3'+ save_name)
            ageutils.save_checkpoint({'epoch': epoch,
                             'state_dict': bridgemodel.state_dict(), 'optimizer': optimizer.state_dict()}, 
                             '1'+ save_name) #by DG
            
                        
        #accuracy = ageutils.mytest(test_loader, dresmodel, lightcnnmodel, device)
        #print('test accuracy is :', accuracy)
        

if __name__ == '__main__':
    settings.init()
    main()
