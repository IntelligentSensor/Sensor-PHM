//
//  main.c
//  corr
//
//  Created by Tunghao on 2019/4/19.
//  Copyright © 2019年 Tunghao. All rights reserved.
//

#include <stdio.h>

int main(int argc, const char * argv[]) {
    
    // insert code here...
    int i,j;
    float mx,my,sx,sy,sxy,denom,r;
    
    /* Calculate the mean of the two series x[], y[] */
    mx = 0;
    my = 0;
    for (i=0;i<n;i++) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;
    
    /* Calculate the denominator */
    sx = 0;
    sy = 0;
    for (i=0;i<n;i++) {
        sx += (x[i] - mx) * (x[i] - mx);
        sy += (y[i] - my) * (y[i] - my);
    }
    denom = sqrt(sx*sy);
    
    /* Calculate the correlation series */
    for (delay=-maxdelay;delay<maxdelay;delay++) {
        sxy = 0;
        for (i=0;i<n;i++) {
            j = i + delay;
            if (j < 0 || j >= n)
                continue;
            else
                sxy += (x[i] - mx) * (y[j] - my);
            /* Or should it be (?)
             if (j < 0 || j >= n)
             sxy += (x[i] - mx) * (-my);
             else
             sxy += (x[i] - mx) * (y[j] - my);
             */
        }
        r = sxy / denom;
        
        /* r is the correlation coefficient at "delay" */
        
    }
    
    printf("Hello, World!\n");
    return 0;
}
