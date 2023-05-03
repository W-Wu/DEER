'''
DEER
Code for preparing labels

Author:
    Wen Wu 2022
'''

import os
import numpy as np

file_path="MSP-PODCAST-Publish-1.8/label/Labels.txt"
out_dir='msp-data/'
f= open(file_path,'r')

Flag=0
line = f.readline()
label_dic = {}
while line:
    if line.startswith('MSP'):
        if Flag == 1:
            label_all.append(avg)
            label_dic[name]=np.array(label_all)
        label_all=[]
        tmp = line.split(';')
        # print(tmp) #['MSP-PODCAST_0001_0008.wav', ' N', ' A:2.200000', ' V:4.000000', ' D:2.600000', '\n']
        name=tmp[0].strip()
        a=float(tmp[2].split(':')[-1])
        v=float(tmp[3].split(':')[-1])
        d=float(tmp[4].split(':')[-1])
        avg=[v,a,d]
        Flag = 1
    elif line.startswith('WORKER'):
        tmp = line.split(';')
        # print(tmp) #['WORKER0000001', ' Neutral', ' Neutral,Confused', ' A:1.000000', ' V:4.000000', ' D:1.000000', '\n']
        a=float(tmp[-4].split(':')[-1])
        v=float(tmp[-3].split(':')[-1])
        d=float(tmp[-2].split(':')[-1])
        e=[v,a,d]
        label_all.append(e)
    line=f.readline()


label_all.append(avg)
label_dic[name]=np.array(label_all)
assert len(label_dic)==73042,len(label_dic)
np.save(os.path.join(out_dir,'msp-label.npy'),label_dic)
