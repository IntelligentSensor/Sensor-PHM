//
//  Wavelet.c
//  logistic
//
//  Created by Tunghao on 2019/1/29.
//  Copyright © 2019年 Tunghao. All rights reserved.
//
#include "Wavelet.h"

/*****************normalizeVec********************/

void normalizeVec(float* vec, int dim)
{
    int i;
    float sum = 0;
    for(i=0;i<dim;i++)
    {
        sum += vec[i];
        for (i=0; i<dim; i++)
        {
            vec[i]/=sum;
        }
    }
}

/**********1D-Covlution********************************
 *  说明: 循环卷积,卷积结果的长度与输入信号的长度相同
 *
 *  输入参数: data[],输入信号; core[],卷积核; cov[],卷积结果;
 *  n,输入信号长度; m,卷积核长度.
 ******************************************************/

void Covlution(double data[], double core[], double cov[],int n, int m)
{
    int i = 0;
    int j = 0;
    int k = 0;
    
    for(i=0; i<n; i++)
    {
        cov[i] = 0;
    }
    
    //前m/2+1行
    i=0;
    for(j=0;j<m/2;j++,i++)
    {
        for(k=m/2-j;k<m;k++)
        {
            cov[i]+=data[k-(m/2-j)]*core[k];
        }
        
        for(k=n-m/2+j;k<n;k++)
        {
            cov[i]+=data[k]*core[k-(n-m/2+j)];
        }
    }
    
    //中间的n-m行
    for(i=m/2;i<=(n-m)+m/2;i++)
    {
        for(j=0;j<m;j++)
        {
            cov[i]+=data[i-m/2+j]*core[j];
        }
    }
    
    //最后m/2-1行
    i=(n-m)+m/2+1;
    for(j=1;j<m/2;j++,i++)
    {
        for(k=0;k<j;k++)
        {
            cov[i]+=data[k]*core[m-j-k];
        }
        
        for(k=0;k<m-j;k++)
        {
            cov[i]+=core[k]*data[n-(m-j)+k];
        }
    }
}

/*****************1/2subsample********************
 *源是M数组，目标N数组，通过含M*N个元素的数组建立映射关系
 *************************************************/
void Resample(unsigned short *pDst, unsigned destLen,
              unsigned short *pSrc, unsigned srcLen)
{
    for (unsigned indexD = 0; indexD < destLen; indexD++)
    {
        unsigned nCount = 0;
        for (unsigned j = 0; j < srcLen; j++)
        {
            unsigned indexM = indexD*srcLen + j;
            unsigned indexS = indexM / destLen;
            nCount += pSrc[indexS];
        }//end for
        pDst[indexD] = nCount / (float)srcLen;
        
    }//end for
}

/***************************2D-wavelet****************************
 *  说明: 二维小波变换,变换两次
 *
 *  输入参数: input[],输入信号; output[],小波变换结果，包括尺度系数和
 *  小波系数; temp[],存放中间结果;Hi[],Daubechies小波基低通滤波器系数;
 *  Lo[],Daubechies小波基高通滤波器系数;n,输入信号长度;
 *  m,Daubechies小波基紧支集长度.
 *****************************************************************/

void Wavelet2D(double input[], double output[], double temp[],
               double Lo[], double Hi[], int n, int m)
{
    int i = 0;
    
    //        Covlution(input, Hi, temp, n, m);
    //
    //        for(i=0;i<n;i+=2)
    //        {
    //            output[i] = temp[i];
    //        }
    //
    //        Covlution(input, Lo, temp, n, m);
    //
    //        for(i=0;i<n;i+=2)
    //        {
    //            output[i] = temp[i];
    //        }
    
    Covlution(input, Hi, temp, n, m);
    
    for(i=0;i<n;i+=2)
    {
        output[i/2] = temp[i];//尺度系数
    }
    
    
    Covlution(input, Lo, temp, n, m);
    
    for(i=0;i<n;i+=2)
    {
        output[n/2+i/2] = temp[i];//小波系数
    }
    
}
