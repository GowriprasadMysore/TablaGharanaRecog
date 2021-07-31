import numpy as np
import torch.utils.data as data_utils
import h5py
from dataLoader import H5_Dataloader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_em import CNNLSTM

def parseArgs(argv):
  parser = argparse.ArgumentParser(description='Trainer')
  parser.add_argument('--debug',action='store_true',help='Debug')
  parser.add_argument('--train_h5_file',type=str,default='train.h5',help='Name of the training data h5')
  parser.add_argument('--dev_h5_file',type=str,default='dev.h5',help='Name of the development data h5')
  parser.add_argument('--test_h5_file',type=str,default='test.h5',help='Name of the test data h5')
  parser.add_argument('--batch_size',type=int,default=16,help='Batch size')
  parser.add_argument('--num_epochs',type=int,default=100,help='Number of epochs to train') 
  parser.add_argument('--inp_channels',type=int,default=1,help='Number of input channels')
  parser.add_argument('--num_targets',type=int,default=6,help='Number of outputs')
  parser.add_argument('--save_step',type=int,default=10,help='Frequency of step to save the model')
  parser.add_argument('--train_loss_log', type=argparse.FileType('w',encoding='utf-8'),default='train_loss_v1.log',help='Training loss log file')
  parser.add_argument('--dev_loss_log', type=argparse.FileType('w',encoding='utf-8'),default='dev_loss_v1.log',help='Development loss log file')
  parser.add_argument('--model_save_dir',type=str,default='models',help='Directory for model save')
  parser.add_argument('--lr_rate',type=float,default=0.01,help="Learning rate for training")
  parser.add_argument('--patience',type=int, default=1,help='Number of patient epochs to reduce the lr rate')
  args=parser.parse_args(argv)
  return args

def train_step(train_loader,model,optimizer,criterion):
  model.train()
  criterion.train()
  train_loss=0
  #outer = tqdm(total=len(train_loader), desc='Epoch', position=0)
  for step,data in enumerate(train_loader):
   batch_data,batch_labels=data
   batch_data=batch_data.unsqueeze(1).cuda(non_blocking=True)
   batch_labels=batch_labels.long().cuda(non_blocking=True)
   pred_labels,_=model(batch_data)
   #print(pred_labels)
   optimizer.zero_grad()
   loss=criterion(pred_labels,batch_labels)
   train_loss=train_loss+loss.item()
   #print(train_loss/((step+1)*8))
   loss.backward()
   optimizer.step()
   #outer.update(1)
  return train_loss*1.0/len(train_loader)
def validation(dev_loader,model,optimizer,criterion):
  model.eval()
  criterion.eval()
  dev_loss=0
  for step1,data in enumerate(dev_loader):
   batch_data,batch_labels=data
   batch_data=batch_data.unsqueeze(1).cuda(non_blocking=True)
   batch_labels=batch_labels.long().cuda(non_blocking=True)
   with torch.no_grad():
    pred_labels,_=model(batch_data)
    loss=criterion(pred_labels,batch_labels)
   dev_loss=dev_loss+loss.item()
  return dev_loss*1.0/len(dev_loader)
def run(train_obj,dev_obj,model,optimizer,criterion,scheduler,args):
 print("Training for {} Epochs".format(args.num_epochs))
 best_dev_loss=100000000
 best_state_dict=None
 start_time=time.time()
 for epoch in range(args.num_epochs):
  model_state_dict=model.module.state_dict()
  print("Running the Epoch :{}".format(epoch))

  train_loader = data_utils.DataLoader(train_obj,batch_size=args.batch_size,shuffle=True)

  dev_loader=data_utils.DataLoader(dev_obj,batch_size=args.batch_size,shuffle=False)
 
  train_loss=train_step(train_loader,model,optimizer,criterion)

  dev_loss=validation(dev_loader,model,optimizer,criterion)

  print(f'Ran {epoch + 1} epochs '
              f'in {time.time() - start_time:.2f} seconds')

  print(f'Training Loss : {train_loss}',f'Development Loss : {dev_loss}')

  if dev_loss < best_dev_loss :
      best_dev_loss=dev_loss
  else:
     print("Loaded Previous model")
     torch.save(model_state_dict, args.model_save_dir+"/model_"+str(epoch-1)+".pt")
     #model.load_state_dict(model_state_dict)
     model.module.load_state_dict(model_state_dict)
  args.train_loss_log.write(str(epoch)+"\t"+str(train_loss)+"\n")

  args.dev_loss_log.write(str(epoch)+"\t"+str(dev_loss)+"\n")

  scheduler.step(dev_loss)
def main(argv):
 args=parseArgs(argv)
 print(args)
 ########### Data Objects #####################
 train_obj=H5_Dataloader(args.train_h5_file)
 
 dev_obj=H5_Dataloader(args.dev_h5_file)
 
 test_obj=H5_Dataloader(args.test_h5_file)
 
 #train_loader = data_utils.DataLoader(train_obj,batch_size=args.batch_size,shuffle=True)
 
 #Model and optimizer
 model=CNNLSTM(args.batch_size,args.inp_channels,args.num_targets)
 
 model=nn.DataParallel(model)
 
 model=model.cuda()
 
 criterion=nn.NLLLoss()

 optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate,weight_decay=0.0000001,betas=(0.98, 0.98),eps=1e-08,amsgrad=False)
 
 # Scheduler
 scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, verbose=True)
 # Running 
 run(train_obj,dev_obj,model,optimizer,criterion,scheduler,args)
if __name__=="__main__":
  torch.multiprocessing.set_start_method('spawn')
  argv=sys.argv[1:]
  main(argv)
