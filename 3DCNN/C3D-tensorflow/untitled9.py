import numpy as np


lst = [6, 3, 4, 9, 2, 5]

for j in range(1, len(lst)):
    key = lst[j]
    i = j - 1
    while i > 0 and lst[i] > key:
        lst[i + 1] = lst[i]
        i -= 1
    lst[i+1] = key

print(lst)
        