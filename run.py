# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:18:40 2021

@author: li
"""

import torch, os
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pandas as pd
import timm
import collections


img_path = "/work/data/leafs-test-dataset"
res_dir = '/work/output/result.csv'
dicts = {0:	'healthy',1:'frog_eye_leaf_spot',	2: 'scab'}
def DataParallel2CPU(model, pth_file):
    state_dict = torch.load(pth_file, map_location='cpu')['state_dict']	# 加载参数
    new_state_dict = collections.OrderedDict()	# 新建字典
    for k, v in state_dict.items():	# 遍历参数，并获取名和值
        if k[:7] == "module.":	# 如果名符合匹配，则截取后面的字符串作为新名字
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)	# 此时，"module."该前缀被清理掉了
    return model

device = torch.device("cpu")
model = timm.create_model('swinv2_tiny_window8_256', pretrained=False, num_classes=3)
model = DataParallel2CPU(model,'./model/best_w.pth')
#model.load_state_dict(torch.load('./model/best_w.pth', map_location='cpu')['state_dict'])

model.eval()
tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),])

data = pd.DataFrame()
test = os.listdir(img_path)

res = pd.DataFrame()
with torch.no_grad():

    all_true = []
    for i in tqdm(test):
        inputs = Image.open(os.path.join(img_path,i)).convert('RGB')   
        inputs = tf(inputs).unsqueeze(0).to(device)
        output = model(inputs)
        output = F.softmax(output, dim=1)
        output = int(torch.argmax(output,dim=1).squeeze().cpu().numpy())
        all_true.append(dicts[output])

res['uuid']=test
res['label']=all_true
res.to_csv(res_dir,index = False)
