import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import random
import pandas as pd
np.random.seed(seed=2)

lable = pd.DataFrame()
data = []
label = []
dicts = {'healthy':	0,'frog_eye_leaf_spot':1,'scab'	:2}



for i in dicts.items():
    dir = "../Ndata/"+i[0] +"/"
    path = os.listdir(dir)
    for p in path:
        data.append(dir+p)
        label.append(i[1])


lable['img'] = data
lable['label'] = label

lable = shuffle(lable)
lable.to_csv("all_data.csv",index = None)

tain_size = 0.85
train = lable.iloc[:int(tain_size*len(lable)),:]
val = lable.iloc[int(tain_size*len(lable)):,:]
train.to_csv("train_data.csv",index = None)
val.to_csv("val_data.csv",index = None)
