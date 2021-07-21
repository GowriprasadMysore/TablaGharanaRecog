import os
import numpy as np
from scipy.io import wavfile
from numpy import savez_compressed
import h5py
import random
import librosa
from sklearn.preprocessing import StandardScaler
from pickle import dump
def main(args):
    data_list=[]
    scaler=StandardScaler()
    for dir1 in args.dirs:
        print(dir1)
        for wav in os.listdir(dir1):
            print(wav)
            data,sr=librosa.load(os.path.join(dir1,wav),sr=16000)
            scaler.partial_fit(data.reshape(-1,1))
    #print("Shape",data.shape)
    #print(data[0])
    #scaler.fit(data)
    dump(scaler, open('scaler_sp.pkl', 'wb'))
if __name__=="__main__":	
    import argparse
    parser=argparse.ArgumentParser(description='Data normalisation Scaler')
    parser.add_argument('--dirs',nargs='+', default=['Ajrada_Gharana','Banaras_Gharana','Delhi_Gharana','Farukhabad_Gharana','Lucknow_Gharana','Punjab_Gharana'])
    parser.add_argument('--output_dir',type=str,default='output')
    args=parser.parse_args()
    print(args.dirs)
    main(args)
