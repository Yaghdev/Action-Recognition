file = open("/home/ankur/data/new_disk/C3D-tensorflow/predict_new.txt")
list_1 = [i for i in file]

right = 0
for  x in list_1:
    print()
    if str(x.split(", ")[0]) == str(x.split(", ")[2]):
        right += 1

print(right/len(list_1))
    
