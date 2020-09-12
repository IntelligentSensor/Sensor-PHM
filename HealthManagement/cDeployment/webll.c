//
//  main.c
//  webll
//
//  Created by Tunghao on 2019/3/26.
//  Copyright © 2019年 Tunghao. All rights reserved.
//

#include <stdio.h>
#include <math.h>

int main(int argc, const char * argv[])
    // insert code here...
    {
        //发生故障的时间
        double time[] = {0.1100,    0.1430,0.1468,0.1939,    0.2509,    0.2717,    0.2823,0.3012,    0.3162,    0.3164,    0.3182,
            0.3398,    0.3471,    0.3493,    0.3639,    0.3892,    0.4114,    0.4569,    0.4651,    0.4710,    0.4774,    0.4997,
            0.5437,    0.5606,    0.5717,    0.5792,    0.5820,    0.5838,    0.5987,    0.6213,    0.6220,    0.6519,    0.6630,
            0.6867,    0.6885,    0.6941,    0.7032,    0.7114,    0.7207,    0.7298,    0.7398,    0.7419,    0.7730,    0.7770,
            0.7934,    0.8073,    0.8231,    0.8320,    0.8325,    0.8396,    0.8447,    0.8459,    0.8463,    0.8568,    0.8676,
            0.8678,    0.8826,    0.9075,    0.9115,    0.9184,    0.9261,    0.9459,    0.9631,    0.9855,    1.0006,    1.0236,
            1.0453,    1.0470,    1.0692,    1.0806,    1.0954,    1.1023,    1.1565,    1.1570,    1.1575,    1.1638,    1.1947,
            1.1994,    1.2094,    1.2114,    1.2165,    1.2197,    1.2273,    1.2536,    1.2730,    1.2790,    1.2994,    1.3033,
            1.3296,    1.3357,    1.4110,    1.4636,    1.4823,    1.4957,    1.5217,    1.5681,    1.8103,    1.8678,    1.8796,
            1.8846};
        //100个故障点
        int num_point = 100;
        
        //下面计算文档中对应的公式
        double xmean = 0.0;
        double ymean = 0.0;
        double beta1 = 0.0;
        double beta21 = 0.0;
        double beta22 = 0.0;
        double beta3 = 0.0;
        double beta4 = 0.0;
        double beta2 = 0.0;
        
        for(int i = 0; i < num_point; i++)
        {
            xmean += log(log(1.0/(1-(i+1)*1.0/(num_point+1))));
            ymean += log(time[i]);
            beta1 += log(time[i])*log(log(1.0/(1-(i+1)*1.0/(num_point+1))));
            beta21 += log(log(1.0/(1-(i+1)*1.0/(num_point+1))));
            beta22 += log(time[i]);
            beta3 += log(time[i])*log(time[i]);
            beta4 += log(time[i]);
        }
        ymean = ymean/num_point;
        xmean = xmean/num_point;
        beta1 = beta1 * num_point;
        beta2 = beta21*beta22;
        beta3 = beta3*num_point;
        beta4 = beta4*beta4;
        double beta = (beta1-beta2)/(beta3-beta4);
        double eta = exp(ymean - xmean/beta);
        return 0;

    printf("Hello, World!\n");
    return 0;
}
