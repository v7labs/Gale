#include <float.h>
/*
* Author:
* Oren Freifeld, freifeld@csail.mit.edu
*/
__global__  void honeycomb( int* seg, 
                            double* centers,
                            int K,
                            int nPts, int xdim, int ydim
                            ){      
    int idx = threadIdx.x + blockIdx.x*blockDim.x; 
    if (idx>=nPts)
        return;       
    
    int x = idx % xdim;
    int y = idx / xdim;   
    
    double dx,dy;
    double D2 = DBL_MAX;
    //double D2 = (xdim*xdim+ydim*ydim )*10000000; // some large number
    
    double d2;
    for (int j=0; j < K;  j++){
        
        dx = (x - centers[j*2+0]);
        dy = (y - centers[j*2+1]);
        d2 = dx*dx + dy*dy;
        if ( d2 <= D2){
              D2 = d2;  
              seg[idx]=j;
        }           
    } 
    return;        
}

