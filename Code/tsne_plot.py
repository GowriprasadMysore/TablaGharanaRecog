import numpy as np
from sklearn.utils import shuffle
from tsne import *
import pylab
import pickle

X=np.load('models_sp_15s/test_sp_15_embeddings.npy')
Y=np.load('models_sp_15s/test_sp_15_labels.npy')


x,y=shuffle(X,Y)
#x=x.reshape(-1,128)
z = tsne(x, 2, 8, 80.0)
np.save('z',z)
np.save('y',y)

list1=[]
a=[0,1,2,3,4,5] ## i have 6 classes. if u have less, remove extra.
for i in range(len(a)):
        c=[]
        for j in range(len(y)):
                if a[i]==y[j]:
                        c.append(z[j])
        list1.append(c)
        

with open('embeddings_sp_15.pkl', 'wb') as handle:
    pickle.dump(list1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
colors = ['r', 'g', 'b', 'c', 'm', 'k','y'] ## i have 7 classes--> 7 colours. if u have less, remove extra
kk=[ 'Ajrada_Gharana','Benaras_Gharana','Dilli_Gharana','Furrukabad_Gharana','Luckhnow_Gharana','Punjab_Gharana'] ## cluster labels
for i in range(len(a)):
        pylab.scatter(np.array(list1[i])[:,0],np.array(list1[i])[:,1],60,c=colors[i],label=kk[i])

pylab.legend()
pylab.show()
pylab.savefig("Embeddings_sp_15.png", dpi=150)
