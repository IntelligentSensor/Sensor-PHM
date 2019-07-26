#-*- coding: UTF-8 -*-

#环境压力指数ESI（0～1）

def ESI(self, Temp):
    if Temp < 0:
        Temp = 0
    elif Temp > 50:
        Temp = 50
        
    ESI = abs(Temp - TempRef)/TempDev
    
    if ESI <= 0.2:
        return low
    elif 0.2 < ESI <= 0.6:
        return middle
    else:
        return high

if __name__ == '__main__':
    Temp = 25
    TempRef = 25
    TempDev = 25