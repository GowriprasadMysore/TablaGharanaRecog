import h5py
import numpy as np
def window(audio,window_size,stride_size):
    start=0
    while start < len(audio):
     yield start,start+window_size
     start=start+stride_size
with h5py.File('test_10s.h5','r') as f:
    test_3sec_h5=h5py.File('test_3sec.h5','w')
    test_3sec_indx=0
    test_5sec_h5=h5py.File('test_5sec.h5','w')
    test_5sec_indx=0
    for key in f.keys():
        data=f[key][:-1]
        label=f[key][-1]
        for slice_indices in window(data,48000,48000):
                if len(data[slice_indices[0]:slice_indices[1]]) == 48000:
                 sample=data[slice_indices[0]:slice_indices[1]]
                 data_point=np.array(list(sample)+[label])
                 test_3sec_h5.create_dataset(str(test_3sec_indx), data=data_point)
                 test_3sec_indx+=1
        for slice_indices in window(data,80000,80000):
                if len(data[slice_indices[0]:slice_indices[1]]) == 80000:
                 sample=data[slice_indices[0]:slice_indices[1]]
                 data_point=np.array(list(sample)+[label])
                 test_5sec_h5.create_dataset(str(test_5sec_indx), data=data_point)
                 test_5sec_indx+=1

    test_3sec_h5.close()
    test_5sec_h5.close()

