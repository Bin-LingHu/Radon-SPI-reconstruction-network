import os
import sys
import shutil

filepath = '../all'

file_dic={}
img_files = os.listdir(filepath)
os.makedirs(filepath+'_half', exist_ok=True)

for img_file in img_files:
    if img_file.split('_')[0] not in file_dic.keys():
        file_dic[img_file.split('_')[0]]=[]
        file_dic[img_file.split('_')[0]].append(img_file)
    else:
        file_dic[img_file.split('_')[0]].append(img_file)

print(file_dic)
for key in file_dic.keys():
    i = 0
    for i in range(0, int(len(file_dic[key])/16+1)):
    #if i  <(len(file_dic[key])/2+1):
        shutil.copytree(os.path.join(filepath, file_dic[key][i]), os.path.join(filepath+'_half', file_dic[key][i]))
        i+=1

