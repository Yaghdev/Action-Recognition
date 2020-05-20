import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open('C:/Users/Krishan/Downloads/ankur1.txt', 'r') as f2:
    mylist = [line.rstrip('\n').split("acc: ")[1] for line in f2]
plt.figure(figsize=(15,7))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
t = np.linspace(0, 1, len(mylist)) 
plt.plot([i for i in range(0, len(mylist))], mylist)



with open('C:/Users/Krishan/Downloads/ankur1.txt', 'r') as f2:
    mylist = [float(line.rstrip('\n').split(" - ")[2][6:]) for line in f2]
plt.figure(figsize=(15,7))
print(mylist)
plt.ylabel('Loss')
plt.xlabel('Epoch')
t = np.linspace(0, 1, len(mylist)) 
plt.plot([i for i in range(0, len(mylist))], mylist)
