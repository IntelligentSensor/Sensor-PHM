//
//  main.c
//  LR
//
//  Created by Tunghao on 2019/3/29.
//  Copyright © 2019年 Tunghao. All rights reserved.
//

#include <stdio.h>
#include "Wavelet.h"
#include "GradDescent.h"
#include "LR.h"

double featVec[24][20] = {0};

int main(int argc, const char * argv[]) {
    
    //Feature extraction
    for(int j = 0; j < 24; j++) //24组数据依次输入
    {
        //一维小波变换
        Wavelet2D(Dataset[j], data_output, temp, Hi, Lo, n, m);
//        printf("\n%d\n", j);
        //    //一维小波变换后的结果写入txt文件
        //    fp=fopen("data_output.txt","w");
        
        //打印一维小波变换后的结果
        for(int i = 0; i < n; i++)
        {
//            printf("%f", data_output[i]);//输出每组数据小波变换的结果
            featVec[j][i] = data_output[i];
            //        fprintf(fp,"%f/n", data_output[i]);
        }
//        normalizeVec(featVec[j], 20);
    }
    
    //Training model
    Traning(BATCH,DIMEN,featVec,target,ITERTION);
    
    //关闭文件
    //    fclose(fp);
    
    return 0;
}
