# Load 모델 (modelcheckpoint) & Predict

from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

### predict 이미지 불러오기
pics_dir = './miniprojectdata/data/predicts'
image_w = 64
image_h = 64
pixels = image_w * image_h * 3

### pred 이미지를 Data 변환
X = []
filenames = []

files = glob.glob(pics_dir + '/*.*')
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

x_pred = np.array(X)
### modelcheckpint Load
model = load_model('./miniprojectdata/checkpoint/cp-05-0.5209.hdf5')
### 예측
y_pred = model.predict(x_pred)
np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})

print(y_pred)
print(y_pred.shape)
cnt = 0

for i in y_pred:
    pre_ans = i.argmax() # 예측 레이블
    pre_ans_str = ''
    if pre_ans == 0: pre_ans_str = '( 강동원 )'
    elif pre_ans == 1: pre_ans_str = '( 강호동 )'
    else: pre_ans_str = '( 강소라 )'
    if i[0] >= 0.8 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[1] >= 0.8 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[2] >= 0.8 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    cnt += 1