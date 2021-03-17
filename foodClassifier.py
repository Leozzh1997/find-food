#%%
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
#%%
#读取图片
def readfile(path,lable):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir),128,128,3), dtype = np.uint8)
    y = np.zeros(len(image_dir), dtype = np.uint8)
    for i,file in enumerate(image_dir):
         img = cv2.imread(os.path.join(path,file))
         x[i, : , :] = cv2.resize(img,(128,128))
         if lable:
             y[i] = int(file[0])
    if lable:
        return x,y
    else:
        return x

path = "./food-11"
train_x,train_y = readfile(os.path.join(path,"training"),True)
valida_x,valida_y = readfile(os.path.join(path,"validation"),True)
test_x = readfile(os.path.join(path,"testing"),False)


#%%
#图片格式转换
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(21),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

class Img_dataset(Dataset):
    def __init__(self, x, y = None, transform = None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X,Y
        return X

bath_size = 50        #out of memery !!
TrainSet = Img_dataset(train_x,train_y,train_transform)
ValidaSet = Img_dataset(valida_x,valida_y,test_transform)
TrainLoader = DataLoader(TrainSet, bath_size, shuffle = True)
ValidaLoader = DataLoader(ValidaSet, bath_size, shuffle = False)
#%%
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(128,256,3,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )

        self.fc = nn.Sequential(
            nn.Linear(4608,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0],-1)
        return self.fc(out)

#%% training
model = Classifier().cuda()
loss = nn.CrossEntropyLoss()
optimazer = torch.optim.Adam(model.parameters())
num_epoch = 20
for epoch in range(num_epoch):
    epoch_time = time.time()
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    model.train()
    for i,data in enumerate(TrainLoader):
        optimazer.zero_grad()
        train_pre = model((data[0].cuda()))
        batch_loss = loss(train_pre,data[1].cuda())
        batch_loss.backward()
        optimazer.step()

        train_acc += np.sum(np.argmax(train_pre.cpu().data.numpy(),axis = 1) == data[1].numpy())
        train_loss += batch_loss.item()
    '''model.eval()
    with torch.no_grad():
        for i, data in enumerate(ValidaLoader):
            val_pre = model(data[0].cuda())
            batch_loss = loss(val_pre, data[1].cuda())
            val_acc += np.sum(np.argmax(val_pre.cpu().data.numpy(),axis = 1) == data[1].numpy())
            val_loss += batch_loss.item()'''
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f ' % \
          (epoch + 1, num_epoch, time.time() - epoch_time,
           train_acc / TrainSet.__len__(), train_loss / TrainSet.__len__()))


#train+val
#%%
print("\nTrain+V\n")
TrainVSet_X = np.concatenate((train_x,valida_x),axis = 0)
TrainVSet_Y = np.concatenate((train_y,valida_y),axis = 0)
TrainVSet = Img_dataset(TrainVSet_X, TrainVSet_Y, transform = train_transform)
TrainVLoader = DataLoader(TrainVSet, bath_size, shuffle = True)

model_pro = model
loss = nn.CrossEntropyLoss()
optimazer = torch.optim.Adam(model_pro.parameters())
num_epoch = 10
for epoch in range(num_epoch):
    model_loss = 0
    model_acc = 0
    epoch_time = time.time()
    model_pro.train()
    for i, data in enumerate(TrainVLoader):
        optimazer.zero_grad()
        pre_y = model_pro(data[0].cuda())
        batch_loss = loss(pre_y,data[1].cuda())
        batch_loss.backward()
        optimazer.step()

        model_loss += batch_loss.item()
        model_acc += np.sum(np.argmax(pre_y.cpu().data.numpy(), axis = 1) == data[1].numpy())

    print('[%d/%d] time:%.2f(s) acc:%.2f loss:%.2f' % (epoch+1,num_epoch,time.time() - epoch_time,
          model_acc / TrainVSet.__len__(), model_loss / TrainVSet.__len__()))
#%%
torch.save(model_pro, 'mode_classfier.pkl')
#%%
model_pro = torch.load('mode_classfier.pkl')
#%% testing
model_pro = model_pro.cuda()
test_set = Img_dataset(test_x,None,test_transform)
test_loader = DataLoader(test_set, bath_size, shuffle = False)
model_pro.eval()
predic = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        pre_y = model_pro(data.cuda())
        test_lable = np.argmax(pre_y.cpu().data.numpy(),axis = 1)
        for y in test_lable:
            predic.append(y)

with open("predict.csv",'w') as f:
    f.write('ID,lable\n')
    for i,y in enumerate(predic):
        f.write('{},{}\n'.format(i,y))












