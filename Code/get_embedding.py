import numpy as np
import torch,sys,argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from model_em import CNNLSTM
import h5py
from dataLoader import H5_Dataloader
import torch.utils.data as data_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def parseArgs(argv):
  parser = argparse.ArgumentParser(description='Trainer')
  parser.add_argument('--debug',action='store_true',help='Debug')
  parser.add_argument('--dev_h5_file',type=str,default='test.h5',help='Name of the development data h5')
  parser.add_argument('--batch_size',type=int,default=16,help='Batch size')
  parser.add_argument('--inp_channels',type=int,default=1,help='Number of input channels')
  parser.add_argument('--num_targets',type=int,default=6,help='Number of outputs')
  parser.add_argument('--checkpoint',type=str,default='models/model.pt',help='Model best check point to load and ')
  parser.add_argument('--emb_file_name', type=str,default='models/test_embeddings.npy',help='file for storing the embeddings')

  parser.add_argument('--labels_name',type=str,default='models/test_labels.npy',help='File for saving the labels')
  args=parser.parse_args(argv)
  return args
def main(argv):
  args=parseArgs(argv)
  ########### Data Objects #####################
 
  dev_obj=H5_Dataloader(args.dev_h5_file)
  ## development and test loader 
  dev_loader = data_utils.DataLoader(dev_obj,batch_size=args.batch_size,shuffle=False)
  print(len(dev_loader))
  use_cuda = True
  DEVICE = torch.device('cuda' if use_cuda else 'cpu')   # 'cpu' in this case
  model=CNNLSTM(args.batch_size,args.inp_channels,args.num_targets)
  
  #model=nn.DataParallel(model)
  print(DEVICE)
  model.to(DEVICE)
  #model=model.cuda()
  ## Load the statedict
  #from collections import OrderedDict
  #new_state_dict = OrderedDict()
  #state_dict = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
  #for k, v in state_dict.items():
  #  name = k[7:] # remove `module.`
  #  new_state_dict[name] = v
  model.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc: storage))
  #model.load_state_dict(new_state_dict)
  criterion=nn.NLLLoss()
  # Extracting the embeddings
  model.eval()
  criterion.eval()
  embedding_list=[]
  labels_list=[]
  for step1,data in enumerate(dev_loader):
   batch_data,batch_labels=data
   batch_data=batch_data.unsqueeze(1).to(DEVICE)
   with torch.no_grad():
    #print(batch_data)
    _,embeddings=model(batch_data)
    embeddings=embeddings.flatten().cpu().numpy().reshape(len(batch_data),-1)
    print(len(embeddings))
    for indx,_ in enumerate(embeddings):
       embedding_list.append(embeddings[indx])
       labels_list.append(batch_labels[indx])
  np.save(args.emb_file_name,np.array(embedding_list))
  np.save(args.labels_name,np.array(labels_list).reshape(-1,1))
if __name__=="__main__":
  argv=sys.argv[1:]
  main(argv)
