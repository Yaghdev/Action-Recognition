import shutil
import os
path = "/home/ankur/data/new_disk/3DCnn-output/"
for i in os.listdir(path):
    if(i.split(".")[0]=='checkpoint'):
        continue
    if not os.path.exists(path+i.split(".")[0]):
        os.makedirs(path+i.split(".")[0])
    
        shutil.move(path+i.split(".")[0]+'.data-00000-of-00001', path+i.split(".")[0])
        shutil.move(path+i.split(".")[0]+'.meta', path+i.split(".")[0])
        shutil.move(path+i.split(".")[0]+'.index', path+i.split(".")[0])
