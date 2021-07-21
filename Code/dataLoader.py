import torch.utils.data as data_utils
import torch
import h5py
class H5_Dataloader(data_utils.Dataset):
    def __init__(self,raw_file):
        self.raw_file = raw_file 
        self.h5=h5py.File(self.raw_file,'r')
        self.keys=list(self.h5.keys())
    def __len__(self):
        """Denotes the total number of samples
        """
        return len(self.keys)

    def __getitem__(self, indx):
        key=self.keys[indx]
        return [torch.from_numpy(self.h5[key][:-1]).float(),self.h5[key][-1]]
'''data_set=H5_Dataloader('test.h5')
loader=data_utils.DataLoader(data_set,batch_size=8,shuffle=False)
for step,data in enumerate(loader):
 print(data[0])'''

    
