import cv2
import pygame
import numpy as np
from pygame.locals import *
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

neural=MLPClassifier(hidden_layer_sizes=(100,100),activation='logistic',alpha=0.01)

data,targets=[],[]
def collect_data(img,value):
    global data,targets
    sample_data=[]
    img_h,img_w,_=img.shape
    for y in range(im_h):
        for x in range(im_w):
            col_org=img[y][x][::-1]
            sample_data.extend(col_org)

    sample_data=np.array(sample_data).reshape(1,-1)
    sample_data=np.divide(sample_data,255.0)
    data.append(sample_data)
    targets.append(value)

def trainer():
    global data,target
    data_train=np.array(data).reshape(len(data),-1)
    targets_train=np.array(targets).reshape(-1,1)
    print(data_train.shape,targets_train.shape)
    neural.fit(data_train,targets_train)

def classify():
    new_data=[]
    img_h,img_w,_=img.shape
    for y in range(im_h):
        for x in range(im_w):
            col_org=img[y][x][::-1]
            new_data.extend(col_org)
    new_data=np.array(new_data).reshape(1,-1)
    new_data=np.divide(new_data,255.0)
    pre=neural.predict(new_data)
    print(pre)

def save_model():
    global neural
    joblib.dump(neural,'')

w,h=600,400
scale=40
pygame.init()
screen=pygame.display.set_mode((w,h),0,32,pygame.HWSURFACE)
screen.fill((255,255,255))

cap=cv2.VideoCapture(0)
start=False
input_array=[]
labels=[]
while True:
    for e in pygame.event.get():
        if e.type==KEYDOWN:
            if e.key==K_q:
                pygame.quit()
            if e.key==K_s:
                start=True
            if e.key==K_p or e.key==K_n:
                if e.key==K_p:
                    target=1
                else:
                    target=0
                collect_data(img,target)
            if e.key==K_t:
                trainer()
            if e.key==K_c:
                classify()
            if e.key==K_d:
                save_model()

    if start:
        _,img=cap.read()
        img=cv2.resize(img,(w//scale,h//scale))
        cv2.imshow("cv",img)
        if cv2.waitKey(1)==ord('q'):
            break
        im_h,im_w,_=img.shape

        for y in range(im_h):
            for x in range(im_w):
                col_org=img[y][x][::-1]
                pygame.draw.rect(screen,col_org,(x*scale,y*scale,scale,scale))

    pygame.display.update()
