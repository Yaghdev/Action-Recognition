

from random import shuffle
path = "D:/autism/3DCNN-master/C3D-tensorflow/list/final_train_3.list"
path_2 = "D:/autism/3DCNN-master/C3D-tensorflow/list/final_train_3.list"
file = open(path)
lst = [i.rstrip("\n") for i in file]
shuffle(lst)
file_1 = open(path_2, 'w')
for i in lst:
    try:        
        if str(i.split("/")[5]) == str('jpegs_256'):
            srr = i.split("/").pop(6)
            file_1.write(srr+"\n")
    except:
        print("fgvegve")
        

 
    