import numpy as np
l=[]
with open('train.log', 'r+') as f:
    lines = f.readlines()
    for i in range(0, len(lines)):
        if('Validation Data Eval:' in lines[i]):
            l.append((lines[i+1].split(' ')[1]).rstrip())

l=np.asarray(l)
l = list(map(float, l))
print(l)
print(np.max(l))



