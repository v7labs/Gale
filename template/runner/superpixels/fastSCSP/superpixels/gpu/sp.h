
/**
* Authors:
* Oren Freifeld, freifeld@csail.mit.edu
* Yixin Li, Email: liyixin@mit.edu
**/
__device__ inline double calc_squared_mahal_1d(double* val,
                                        const double* mu,
                                        const double* J
                                        ){
	  // val: value of interest									
	  // mu: mean
	  // J: inverse of the covariance										
      double J00 = J[0];       
      double x0 = val[0]-mu[0];      
      return x0*x0*J00;  
}

__device__ inline double calc_squared_mahal_2d(double* val,
                                        const double* mu,
                                        const double* J
                                        ){
	  // val: value of interest									
	  // mu: mean
	  // J: inverse of the covariance											
      double J00 = J[0];
      double J01 = J[1];
      double J11 = J[3];   
      
      double x0 = val[0]-mu[0];
      double x1 = val[1]-mu[1];
      
      double res = x0*x0*J00 + x1*x1*J11 + 2*x0*x1*J01;
          
      return res;   
}


__device__ inline double calc_squared_mahal_3d(double* val,
                                        const double* mu,
                                        const double* J
                                        ){
	  // val: value of interest									
	  // mu: mean
	  // J: inverse of the covariance											
      double J00 = J[0];
      double J01 = J[1];
      double J02 = J[2];
      double J11 = J[4];   
      double J12 = J[5]; 
      double J22 = J[8]; 
      
      double x0 = val[0]-mu[0];
      double x1 = val[1]-mu[1];
      double x2 = val[2]-mu[2];
      
      double res = x0*x0*J00 + x1*x1*J11 + x2*x2*J22 +
            2*(x0*x1*J01 + x0*x2*J02 + x1*x2*J12);
                    
      return res;              

}


__device__ inline double calc_squared_eucli_1d(double* val,const double* mu, const int std){
      // val: value of interest                                 
      // mu: mean     
      double x0 = val[0]-mu[0];      
      return pow(x0/double(std),2);  
}


__device__ inline double calc_squared_eucli_2d(double* val,const double* mu, const int std){
      // val: value of interest                                 
      // mu: mean      
      double x0 = val[0]-mu[0];
      double x1 = val[1]-mu[1];
      double res = (x0*x0 + x1*x1)/double(std)/double(std);      
      return res;   
}


__device__ inline double calc_squared_eucli_3d(double* val, const double* mu, const int std){
      // val: value of interest                                 
      // mu: mean
      double x0 = val[0]-mu[0];
      double x1 = val[1]-mu[1];
      double x2 = val[2]-mu[2];     
      double res = (x0*x0 + x1*x1 + x2*x2)/double(std)/double(std);                 
      return res;              
}




//__device__ inline bool ischangbale_by_num(int num){
//    // The string for the condition below was computed in top.py
//    if ((num<0) || (num>255))
//        return 0;
//      
//        
//    return (
//    ((num == 2)||  (num == 3)||  (num == 6)||  (num == 7)||  (num == 8)||  (num == 9)||  (num == 11)||  (num == 15)||  (num == 16)||  (num == 20)||  (num == 22)||  (num == 23)||  (num == 31)||  (num == 40)||  (num == 41)||  (num == 43)||  (num == 47)||  (num == 63)||  (num == 64)||  (num == 96)||  (num == 104)||  (num == 105)||  (num == 107)||  (num == 111)||  (num == 127)||  (num == 144)||  (num == 148)||  (num == 150)||  (num == 151)||  (num == 159)||  (num == 191)||  (num == 192)||  (num == 208)||  (num == 212)||  (num == 214)||  (num == 215)||  (num == 223)||  (num == 224)||  (num == 232)||  (num == 233)||  (num == 235)||  (num == 239)||  (num == 240)||  (num == 244)||  (num == 246)||  (num == 247)||  (num == 248)||  (num == 249)||  (num == 251)||  (num == 252)||  (num == 253)||  (num == 254)||  (num == 255))
//    );
//    }
    
__device__ bool lut[256] = {0,0,1,1,0,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1};
    
    
__device__ inline bool ischangbale_by_nbrs(bool* nbrs){
	// This function does the following:
	// 1) converts the arrray of binary labels of the 8 nbrs into an integer  btwn 0 and 255
	// 2) does a lookup check on the resulting function using the ischangbale_by_num function.
    /*int num=(nbrs[7]+    // SE
             nbrs[6]*2+  // S
             nbrs[5]*4+  // SW
             nbrs[4]*8+  // E
             nbrs[3]*16+ // W
             nbrs[2]*32+ // NE
             nbrs[1]*64+ // N
             nbrs[0]*128); // NW  
    */
    int num = 0;
#pragma unroll
   for (int i=7; i>=0; i--){    
      num <<= 1;
      if (nbrs[i]) num++;
   } 
    if (num == 0)
		return 0;
	else		 
		return lut[num];  
    //return ischangbale_by_num(num);
    }
    
   
/*
* Set the elements in nbrs "array" to 1 if corresponding neighbor pixel has the same superpixel as "label"
*/
__device__ inline void set_nbrs(int idx,int xdim,int ydim,
                                const bool x_greater_than_1,
                                const bool y_greater_than_1,
                                const bool x_smaller_than_xdim_minus_1,
                                const bool y_smaller_than_ydim_minus_1,
                                const int* seg,bool* nbrs,int label){
    // init            
    nbrs[0]=nbrs[1]=nbrs[2]=nbrs[3]=nbrs[4]=nbrs[5]=nbrs[6]=nbrs[7]=0;
    
    if (x_greater_than_1 && y_greater_than_1){// NW        
        nbrs[0] = (label == seg[idx-xdim-1]);  
    }
    if (y_greater_than_1){// N        
        nbrs[1] = (label == seg[idx-xdim]);  
    }
    if (x_smaller_than_xdim_minus_1 && y_greater_than_1){// NE         
        nbrs[2] = (label == seg[idx-xdim+1]);  
    }
    if (x_greater_than_1){// W
        nbrs[3] = (label==seg[idx-1]);        
    }
    if (x_smaller_than_xdim_minus_1){// E
       nbrs[4] = (label==seg[idx+1]);  
    }
    if (x_greater_than_1 && y_smaller_than_ydim_minus_1){// SW
       nbrs[5] = (label==seg[idx+xdim-1]);  
    }
    if (y_smaller_than_ydim_minus_1){// S 
       nbrs[6] = (label==seg[idx+xdim]);  
    }    
    if (x_smaller_than_xdim_minus_1 && y_smaller_than_ydim_minus_1){// SE
       nbrs[7] = (label==seg[idx+xdim+1]);  
    }
    return;
}   


__device__ inline double cal_posterior(bool isValid, int dir,
    const bool calculate_cov, double* imgC, double* pt,
    const double * log_counts, const double prior_weight,   
    const double* mu_i, const double* mu_s,
    const double* J_i, const double* J_s, 
    const double* logdet_Sigma_i, const double* logdet_Sigma_s,
    const int i_std, const int s_std, const bool img_isNaN)
{
      double res = -100000000; // some large negative number  
      if (isValid){

          //use the covariances
          if (calculate_cov){
              
              // calculate for color component is this pixel is not NaN
              if (img_isNaN){
                res = 0.0;

              }else{
 #if NUM_OF_CHANNELS == 1
                res = -calc_squared_mahal_1d(imgC,mu_i,J_i);  
#else
                res = -calc_squared_mahal_3d(imgC,mu_i,J_i);      
#endif      
                res -= logdet_Sigma_i[dir];               
              }

              //space component
              res -= calc_squared_mahal_2d(pt,mu_s,J_s);              
              res -= logdet_Sigma_s[dir]; 

          }else{
#if NUM_OF_CHANNELS == 1
            res = -calc_squared_eucli_1d(imgC,mu_i,i_std);  
#else
            res = -calc_squared_eucli_3d(imgC,mu_i,i_std);      
#endif     
            res -= calc_squared_eucli_2d(pt,mu_s, s_std); 
         }
       
        //add in prior prob
#if USE_COUNTS 
        res *= (1-prior_weight);
        double prior = prior_weight * log_counts[dir];
        res += prior;
#endif

        
      }
      return res;

     }
