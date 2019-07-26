import numpy as np
import pandas as pd

def readfile():
    file_name = 'data/CirculationPoolField.txt'
    circulation = []
    for line in open(file_name , encoding='gbk' , errors='ignore'):
        line = line.split()
        circulation.append(line)   
    
    clis = []
    for i in circulation[2]:
        clis.append(float(i))
    
    clisdiff = np.diff(clis, n=1, axis = -1)

