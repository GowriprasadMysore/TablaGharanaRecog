import numpy as np
import torch,sys,argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from model import CNNLSTM
import h5py
from dataLoader import H5_Dataloader
import torch.utils.data as data_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def parseArgs(argv):
  parser = argparse.ArgumentParser(description='Trainer')
  parser.add_argument('--debug',action='store_true',help='Debug')
  parser.add_argument('--train_h5_file',type=str,default='train.h5',help='Name of the training data h5')
  parser.add_argument('--dev_h5_file',type=str,default='dev.h5',help='Name of the development data h5')
  parser.add_argument('--test_h5_file',type=str,default='test.h5',help='Name of the test data h5')
  parser.add_argument('--batch_size',type=int,default=64,help='Batch size')
  parser.add_argument('--inp_channels',type=int,default=1,help='Number of input channels')
  parser.add_argument('--num_targets',type=int,default=6,help='Number of outputs')
  parser.add_argument('--checkpoint',type=str,default='models/model_0.pt',help='Model best check point to load and test')
  parser.add_argument('--train_loss_log', type=argparse.FileType('r',encoding='utf-8'),default='train_loss.txt',help='Training loss log file for plotting')
  parser.add_argument('--dev_loss_log', type=argparse.FileType('r',encoding='utf-8'),help='Development loss log file for plotting')
  parser.add_argument('--true_pred_labels',type=str,default='true_pred_labels.txt',help="File to store true and pred labels")
  args=parser.parse_args(argv)
  return args
def confusion_matrix_cal(true,pred,labels):
  matrix = confusion_matrix(true,pred, labels=labels)
  #dictionary={"True_positives":tp,"True_negatives":tn,"False_positives":fp,"False_negatives":fn}
  report = classification_report(true,pred,labels=labels)
  return matrix,report
def find_accuracy(loader,model,criterion):
  model.eval()
  criterion.eval()
  total_loss=0
  accuracy=0
  true_labels=[]
  predicted_labels=[]
  for step1,data in enumerate(loader):
   batch_data,batch_labels=data
   batch_data=batch_data.unsqueeze(1).cuda(non_blocking=True)
   batch_labels=batch_labels.long().cuda(non_blocking=True)
   with torch.no_grad():
    pred_labels,_=model(batch_data)
    loss=criterion(pred_labels,batch_labels)
   labels_indices=torch.argmax(pred_labels,dim=1)
   true_labels=true_labels+list(batch_labels.flatten().cpu().numpy())
   predicted_labels=predicted_labels+list(labels_indices.flatten().cpu().numpy())
   accuracy+=torch.sum(torch.eq(batch_labels,labels_indices))*1.0/(batch_labels.shape[0])
   total_loss=total_loss+loss.item()
  accuracy=accuracy/len(loader)
  return total_loss*1.0/len(loader),accuracy,true_labels,predicted_labels
def main(argv):
  args=parseArgs(argv)
  ########### Data Objects #####################
  train_obj=H5_Dataloader(args.train_h5_file)
 
  dev_obj=H5_Dataloader(args.dev_h5_file)
 
  test_obj=H5_Dataloader(args.test_h5_file)
  ## development and test loader 
  dev_loader = data_utils.DataLoader(dev_obj,batch_size=args.batch_size,shuffle=False)

  test_loader = data_utils.DataLoader(test_obj,batch_size=args.batch_size,shuffle=False)
  print(len(dev_loader),len(test_loader))
  model=CNNLSTM(args.batch_size,args.inp_channels,args.num_targets)
  
  model=nn.DataParallel(model)
  
  model=model.cuda()
  
  model.module.load_state_dict(torch.load(args.checkpoint))
  
  criterion=nn.NLLLoss()
  
  loss,accuracy,true_labels,pred_labels=find_accuracy(dev_loader,model,criterion)
  
  print("Development",loss,accuracy)
    
  loss,accuracy,true_labels,pred_labels=find_accuracy(test_loader,model,criterion)
  
  print("Test",loss,accuracy)  
  with open(args.true_pred_labels,'w') as fw:
   for l1,l2 in zip(true_labels,pred_labels):
    fw.write(str(l1)+" "+str(l2)+"\n")
  
  confusion_mat,report = confusion_matrix_cal(true_labels,pred_labels,[5,4,3,2,1,0])

  print("Confusion Matrix")
  print(confusion_mat)
  #print("Measure",dictionary_measures)
  print("Report:\n", report)
  train_loss=[]
  lines=args.train_loss_log.readlines()
  for line in lines:
    train_loss.append(float(line.rstrip().split('\t')[1]))
  dev_loss=[]
  lines=args.dev_loss_log.readlines()
  for line in lines:
    dev_loss.append(float(line.rstrip().split('\t')[1]))
  plt.plot(train_loss,label='Train loss',color='r')
  plt.plot(dev_loss,label='Development loss',color='g')
  plt.legend()
  plt.savefig(sys.argv[3])
  plt.show()
if __name__=="__main__":
  argv=sys.argv[1:]
  main(argv)
