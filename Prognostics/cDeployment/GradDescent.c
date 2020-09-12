//
//  GradientDescent.c
//  logistic
//
//  Created by Tunghao on 2019/3/7.
//  Copyright © 2019年 Tunghao. All rights reserved.
//

#include "GradDescent.h"

void Traning(int Batch, int Dimen, double VectorSet[Batch][Dimen], double TargetSet[], int iteration)
{
    
    double error_sum = 0;
    double theta[20] = {0};  //theta的初值全部设为，从零开始
    
    //    Batchsize = sizeof(VectorSet)/sizeof(VectorSet[0]);
    //    Dimension = sizeof(VectorSet[0])/sizeof(VectorSet[0][0]);
    
    int temp;
    for(temp=1;temp<= iteration;temp++)
        
    {                               //temp用来表示迭代次数，然后i表示样本的数量，每次更新temp，都要迭代一轮i
        int k,i;
        for(i = 0;i< Batch;i++)          //因为训练数据是4个 所以i从0到3
        {
            double h = 0;
            
            for(k = 0;k< Dimen;k++)     //k负责迭代的是x的维数，也就是theta的维数，二维数据要设置三维的维度
            {
                h = h + theta[k]*VectorSet[i][k];     //用累加和求预测函数h（x）并且记住，每次求完h算完残差之后，都要把h重新归零
            }
            error_sum = h - TargetSet[i];
            for(k = 0;k< Dimen;k++)
            {
                theta[k] = theta[k] - 0.04*error_sum*VectorSet[i][k]; //梯度下降，更新k个theta值，并且减去导数方向乘以x乘以学习率alpha
            }
            
            printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                   theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7],theta[8],
                   theta[9],theta[10],theta[11],theta[12],theta[13],theta[14],theta[15],theta[16],
                   theta[17],theta[18],theta[19]);
        }
    }
    
}
