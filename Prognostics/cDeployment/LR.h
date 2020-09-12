//
//  LR.h
//  LR
//
//  Created by Tunghao on 2019/1/29.
//  Copyright © 2019年 Tunghao. All rights reserved.
//

#ifndef LR_h
#define LR_h

static double matrix[6][4] = {
    {1,47,76,24}, //include x0=1
    {1,46,77,23},
    {1,48,74,22},
    {1,34,76,21},
    {1,35,75,24},
    {1,34,77,25},
    
};

static double result[6] = {1,1,1,0,0,0};
static double theta[4] = {2,2,2,2};

extern double sigmoidFunction(double x);


#endif /* LR_h */
