#-*- coding: UTF-8 -*-
import math
import numpy as np
from scipy import log
from scipy.optimize import curve_fit
# from scipy import log as log print pcov

#三次多项式回归 最小二乘拟合
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                                # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)                    # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)             # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)              # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot   #准确率
    return results

#对数回归 指幂数 curve_fit拟合
def func1(x, a, b):
    y = 1 - exp(-pow((x/a),b))   #失效率
    return y

def func2(x, a, b):
    y = a * exp(-b * x)          #指数
    return y

def func3(x, a, b):
    y = a * log(x) + b           #对数
    return y

def curvefit(func, x, y):
    results = {}
    popt, pcov = curve_fit(func, x, y)
    results['polynomial'] = popt

    # r-squared
    yhat = func(x ,popt[0] ,popt[1] )          # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)                    # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)             # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)              # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

#使用math做线性拟合
def linefit(x , y):
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r

if __name__ == '__main__':
    X=[ 1 ,2  ,3 ,4 ,5 ,6]
    Y=[ 2.5 ,3.51 ,4.45 ,5.52 ,6.47 ,7.51]
    a,b,r=linefit(X,Y)
    print("X=",X)
    print("Y=",Y)
    print("拟合结果: y = %10.5f x + %10.5f , r=%10.5f" % (a,b,r) )

    