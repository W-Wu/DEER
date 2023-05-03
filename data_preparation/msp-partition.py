'''
DEER
Code for splitting train/validation/test data.

Author:
    Wen 2022
'''

import numpy as np

file_path="/MSP-PODCAST-Publish-1.8/Partitions.txt"
f= open(file_path,'r')
partition_dic={}
line = f.readline()
while line:
	tmp = line.split(';')	#['Train', ' MSP-PODCAST_0023_0048.wav\n']
	if len(tmp)<2:
		line = f.readline()
		continue
	part = tmp[0]
	name = tmp[1].strip()
	if part in partition_dic:
		partition_dic[part].append(name)
	else:
		partition_dic[part]=[name]
	line = f.readline()
np.save('msp-data/partition_dic.npy',partition_dic)
