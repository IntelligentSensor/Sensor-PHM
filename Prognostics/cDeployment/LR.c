////
////  LR.c
////  LR
////
////  Created by Tunghao on 2019/1/29.
////  Copyright © 2019年 Tunghao. All rights reserved.
////

#include "LR.h"
#include <math.h>

double sigmoidFunction(double x)
{
    double ex;
    ex = pow(2.718281828,x);
    return ex/(1+ex);
}
