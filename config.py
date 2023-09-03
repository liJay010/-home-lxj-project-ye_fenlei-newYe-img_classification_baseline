# -*- coding: utf-8 -*-
import os


class Config:

    root = './'
    model_name = 'modelname'
    stage_epoch = [8,16,24,32,60,72,84,96]
    batch_size = 48
    #label的类别数
    num_classes = 3
    max_epoch = 100
    #保存模型的文件夹
    ckpt = 'ckpt'
    #保存提交文件的文件夹
    sub_dir = 'submit'
    #初始的学习率
    lr = 0.0002
    #保存模型当前epoch的权重
    current_w = 'current_w.pth'
    #保存最佳的权重
    best_w = 'best_w.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 1.2
    #for test
    temp_dir=os.path.join(root,'temp')


config = Config()
