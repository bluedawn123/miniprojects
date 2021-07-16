# 이미지 데이터 처리

from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split


### 이미지 파일 불러오기 및 카테고리 정의
pic_dir = './miniprojectdata/data'
categories = ['kangdongwon', 'kanghodong', 'kangsora']
nb_classes = len(categories)
### 가로, 세로, 채널 쉐이프 정의
image_w = 64
image_h = 64
pixels = image_h * image_w * 3
### 이미지 파일 Data화
X = []
Y = []
for index, things in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[index] = 1
    image_dir = pic_dir + '/' + things        #예를들면, './miniprojectdata/data/kangdongwon 
    files = glob.glob(image_dir + "/*.jpg")   #확장자가 jpg인 모든 이미지를 불러오는 라이브러리
    print(things, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        Y.append(label)
#numpy
x = np.array(X)
y = np.array(Y)

### 데이터 train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
xy = (x_train, x_test, y_train, y_test)
### 데이터 SAVE
print('>>> data 저장중 ...')
np.save('./miniprojectdata/data/datasetNPY/datasets.npy', xy)