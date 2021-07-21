import os
import numpy as np
from scipy.io import wavfile
from numpy import savez_compressed
import h5py
import random
import librosa
from sklearn.preprocessing import StandardScaler
#random.seed(111)
from pickle import load

def window(audio,window_size,stride_size):
    start=0
    while start < len(audio):
     yield start,start+window_size
     start=start+stride_size
def main(args):
    window_size=args.segment_length*16000
    stride_size=args.stride_of_segment*16000
    data_dict={}
    train_indx=0
    dev_indx=0
    test_indx=0
    h5_train=h5py.File('train_10s.h5','w')
    h5_dev=h5py.File('dev_10s.h5','w')
    h5_test=h5py.File('test_10s.h5','w')
    scaler=load(open('scaler.pkl', 'rb'))
    for dir1 in args.dirs:
        label=args.dict_labels[dir1]
        data_list=[]
        for wav in os.listdir(dir1):
            data,sr=librosa.load(os.path.join(dir1,wav),sr=16000)
            data=scaler.transform(data.reshape(-1,1)).reshape(-1)
            for slice_indices in window(data,window_size,stride_size):
                if len(data[slice_indices[0]:slice_indices[1]]) == 160000:
                 data_list.append(data[slice_indices[0]:slice_indices[1]])
        random.shuffle(data_list)
        num_samples=len(data_list)
        for indx,sample in enumerate(data_list):
           data_point=np.array(list(sample)+[label])
           #print(len(data_point),type(data_point),data_point[-1])
           if indx <= int(num_samples*0.7):
              h5_train.create_dataset(str(train_indx), data=data_point)
              train_indx+=1
           elif indx > int(num_samples*0.7) and indx <= int(num_samples*0.85):
              h5_dev.create_dataset(str(dev_indx), data=data_point)
              dev_indx+=1
           else:
              h5_test.create_dataset(str(test_indx), data=data_point)
              test_indx+=1
    h5_train.close()
    h5_test.close()
    h5_dev.close()
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description='Data_preparation')
    parser.add_argument('--dirs',nargs='+', default=['Ajrada_Gharana','Banaras_Gharana','Delhi_Gharana','Farukhabad_Gharana','Lucknow_Gharana','Punjab_Gharana'])
    parser.add_argument('--segment_length',type=int,default=10)
    parser.add_argument('--stride_of_segment',type=int,default=8)
    parser.add_argument('--output_dir',type=str,default='output')
    parser.add_argument('--dict_labels',type=dict,default={'Ajrada_Gharana':0,'Banaras_Gharana':1,'Delhi_Gharana':2,'Farukhabad_Gharana':3,'Lucknow_Gharana':4,'Punjab_Gharana':5})
    args=parser.parse_args()
    print(args.dirs)
    main(args)
