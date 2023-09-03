# -*- coding: utf-8 -*-
import torch, time, os, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from tensorboard_logger import Logger
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import Loader2
from config import config
from torchvision.models import resnet18
from torch.utils.data import DataLoader,random_split
import torch.nn.functional as F
import timm
from torch.autograd import Variable
from sklearn import metrics
import numpy as np
from timm.loss import AsymmetricLossMultiLabel ##需要one hot F.one_hot(targets_a,num_classes=5)
import collections
from timm.data.mixup import Mixup
# loss改动-不平衡（focal loss），AsymmetricLossMultiLabel loss(多分类)
# lr调整策略
# mixup（https://blog.csdn.net/Brikie/article/details/113872771），Manifold Mixup（https://blog.csdn.net/Brikie/article/details/114222605）
# 数据增强策略
# 网络模型改动，effcientnet系列
# gan网络生成数据（https://github.com/lucidrains/stylegan2-pytorch），预训练，fineturn，半监督，图像预处理

class args():
    def __init__(self):
        self.command='train'
        self.ckpt=''
        self.ex=None
        self.resume=False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mixup_fn = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
    prob=0.1, switch_prob=0.5, mode='batch',
    label_smoothing=0.1, num_classes=3)
torch.manual_seed(41)
torch.cuda.manual_seed(41)
args=args()

train_loader  = Loader2(train=True)
val_loader  = Loader2(train=False)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
class FusionModel(torch.nn.Module):
 
    def __init__(self, model1, model2, model3):
 
        super(FusionModel, self).__init__()
 
        self.model1 = model1
 
        self.model2 = model2
 
        self.model3 = model3
 
        self.fc = torch.nn.Linear(9, 3)  # 假设最终输出为单个值
 
    def forward(self, x):
 
        output1 = self.model1(x)
 
        output2 = self.model2(x)
 
        output3 = self.model3(x)
 
        fused_output = torch.cat([output1, output2, output3], dim=1)
 
        fused_output = self.fc(fused_output)
 
        return fused_output
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)

def train_epoch(model, optimizer, criterion, train_dataloader,scheduler, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    predicts = np.array([])
    labels = np.array([])
    for inputs, target in train_dataloader:
        # break
        #index = torch.randperm(inputs.size(0)).cuda()
        inputs = inputs.to(device)
        target = target.to(device)
        
        #lam = np.random.beta(1, 1)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, target, alpha=1.0)
        inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
        
        #mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
        # Mixup loss.    



        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        
        #loss = criterion(output, target)
        #F.one_hot(output,num_classes=9)
        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        #loss = lam * criterion(output, target) + (1 - lam) * criterion(output, target[index])
        #loss = lam * criterion(output, F.one_hot(target,num_classes=25)) + (1 - lam) * criterion(output, F.one_hot( target[index],num_classes=25))
        loss.backward()
        optimizer.step()
        
        loss_meter += loss.item()
        it_count += 1


        output = F.softmax(output, dim=1)
        output = torch.argmax(output,dim=1)
        #Score1=((output.cpu().detach().numpy()==target.cpu().detach().numpy()).sum())/len(output.cpu().detach().numpy())
        #Score2 = metrics.f1_score(target.cpu().detach().numpy(), output.cpu().detach().numpy(),average='micro')
        #Score = (Score1 + Score2) / 2
        #f1_meter += Score
        predicts = np.append(predicts,output.cpu().detach().numpy())
        labels = np.append(labels,target.cpu().detach().numpy())
        scheduler.step() 
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e acc:%.3f" % (it_count, loss.item(), metrics.f1_score(labels,predicts,average='micro')))

    print("lr:",optimizer.state_dict()['param_groups'][0]['lr'])
    #scheduler.step() 
    f1_meter = metrics.f1_score(labels,predicts,average='micro')
    return loss_meter / it_count, f1_meter 

def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    predicts = np.array([])
    labels = np.array([])
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            #loss = criterion(output, target)
            loss = criterion(output, target) 
            loss_meter += loss.item()
            it_count += 1
            output = F.softmax(output, dim=1)
            output = torch.argmax(output,dim=1)

    
            #Score1=((output.cpu().detach().numpy()==target.cpu().detach().numpy()).sum())/len(output.cpu().detach().numpy())
            #Score2 = metrics.f1_score(target.cpu().detach().numpy(), output.cpu().detach().numpy(),average='micro')
            #Score = (Score1 + Score2) / 2
            #f1_meter += Score
            predicts = np.append(predicts,output.cpu().detach().numpy())
            labels = np.append(labels,target.cpu().detach().numpy())

    f1_meter = metrics.f1_score(labels,predicts,average='micro')
    score2 = (labels == predicts).sum() / len(labels)
    return loss_meter / it_count, (f1_meter + score2) / 2

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)

# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(args):

    #model1 = timm.create_model('densenet121', pretrained=True, num_classes=3)
    #model2 = timm.create_model('efficientnet_b4', pretrained=True, num_classes=3)
    #model3 = timm.create_model('swinv2_tiny', pretrained=True, num_classes=3)

    #model = FusionModel(model1, model2, model3)
    #model = timm.create_model('swin_s3_base_224', pretrained=True, num_classes=3)
    #model = timm.create_model('deit3_base_patch16_224', pretrained=True, num_classes=3)
    #model = timm.create_model('deit3_small_patch16_384', pretrained=True, num_classes=3)
    model = timm.create_model('swinv2_tiny_window8_256', pretrained=True, num_classes=3)
    model = torch.nn.DataParallel(model)  
    cudnn.benchmark = True
    model = model.cuda()
    
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt)
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    # data
    #model.module.classifier = nn.Linear(in_features=1792, out_features=3, bias=True)
    #print(model)
    #model = model.cuda()

    train_dataloader = DataLoader(train_loader, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_loader, batch_size=config.batch_size, num_workers=4)
    print("train_datasize", len(train_loader), "val_datasize", len(val_loader))

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr,weight_decay=0.00005)
    num_steps = len(train_dataloader) * config.max_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps,eta_min= 0.00001)
    
    #criterion = AsymmetricLossMultiLabel(gamma_pos=1,gamma_neg=4,eps=1e-8)
    criterion = CrossEntropyLabelSmooth(num_classes=9).cuda()
    #criterion = FocalLoss(num_class = 9)

    # 模型保存文件夹
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    # 从上一个断点，继续训练
    if args.resume:
        if os.path.exists(args.ckpt):  # 这里是存放权重的目录
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            # 如果中断点恰好为转换stage的点
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    logger = Logger(logdir=model_save_dir, flush_secs=2)
    # =========>开始训练<=========
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader,scheduler, show_interval=50)
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)
        print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
              % (epoch, stage, train_loss, train_f1, val_loss, val_f1, print_time_cost(since)))
        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_f1', train_f1, step=epoch)
        logger.log_value('val_loss', val_loss, step=epoch)
        logger.log_value('val_f1', val_f1, step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                 'stage': stage}
        save_ckpt(state, best_f1 < val_f1, model_save_dir)
        best_f1 = max(best_f1, val_f1)
        """if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            adjust_learning_rate(optimizer, lr)"""

if __name__ == '__main__':

    train(args)
