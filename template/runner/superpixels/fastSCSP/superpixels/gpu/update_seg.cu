#include "sp.h"

#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#ifndef BAD_TOPOLOGY_LABEL 
#define BAD_TOPOLOGY_LABEL -2
#endif

#ifndef NUM_OF_CHANNELS 
#define NUM_OF_CHANNELS 3
#endif


#ifndef USE_COUNTS
#define USE_COUNTS 1
#endif


#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

/*
* Authors:
* Oren Freifeld, freifeld@csail.mit.edu
* Yixin Li, Email: liyixin@mit.edu
*/
__global__  void find_border_pixels( int* seg, bool* border, int nPts, int xdim, int ydim, const int single_border){   
    
    int idx = threadIdx.x + blockIdx.x*blockDim.x; 
    if (idx>=nPts)
        return; 

    border[idx]=0;  // init        
    
    int x = idx % xdim;
    int y = idx / xdim;        
    int C = seg[idx]; // center 
    int N,S,E,W; // north, south, east,west        
    
    N=S=W=E=OUT_OF_BOUNDS_LABEL; // init 
    
    if (y>1){
        N = seg[idx-xdim]; // above
    }          
    if (x>1){
        W = seg[idx-1];  // left
    }
    if (y<ydim-1){
        S = seg[idx+xdim]; // below
    }   
    if (x<xdim-1){
        E = seg[idx+1];  // right
    }           
    
    // If the nbr is different from the central pixel and is not out-of-bounds,
    // then it is a border pixel.
    if ((N>=0 && C!=N) || (S>=0 && C!=S) || (E>=0 && C!=E) || (W>=0 && C!=W) ){
        if (single_border){
            if (N>=0 && C>N) border[idx]=1; 
            if (S>=0 && C>S) border[idx]=1;
            if (E>=0 && C>E) border[idx]=1;
            if (W>=0 && C>W) border[idx]=1;
        }else{   
            border[idx]=1;  
        }
    }      
    return;        
}




/*
* Update the superpixel labels for pixels 
* that are on the boundary of the superpixels
* and on the (xmod3, ymod3) position of 3*3 block
*/
__global__  void update_seg_subset(
    double* img, int* seg, const bool* border,
    const int * counts, const double * log_counts,
    const double* mu_i, const double* mu_s,
    const double* J_i, const double* J_s, 
    const double* logdet_Sigma_i, const double* logdet_Sigma_s,  
    const int nPts,
    const int xdim, const int ydim,
    const int xmod3, const int ymod3,
    const int nSuperpixels,
    const bool calculate_cov,
    const int s_std, const int i_std,
    const double prior_weight)
{   
    int idx = threadIdx.x + blockIdx.x*blockDim.x; 
    if (idx>=nPts)
        return;

    if (border[idx]==0) return;   

    int x = idx % xdim;  
    if (x % 3 != xmod3) return;  
    int y = idx / xdim;   
    if (y % 3 != ymod3) return;   
    
    const bool x_greater_than_1 = x>1;
    const bool y_greater_than_1 = y>1;
    const bool x_smaller_than_xdim_minus_1 = x<xdim-1;
    const bool y_smaller_than_ydim_minus_1 = y<ydim-1;
    
    int C = seg[idx]; // center 
    int N,S,E,W; // north, south, east,west        

    N=S=W=E=OUT_OF_BOUNDS_LABEL; // init to out-of-bounds 

     
    bool nbrs[8];
        
     
    double* imgC = img + idx * NUM_OF_CHANNELS;
   
    // means
    const double* mu_i_N;
    const double* mu_i_S;
    const double* mu_i_E;
    const double* mu_i_W;
    
    const double* mu_s_N;
    const double* mu_s_S;
    const double* mu_s_E;
    const double* mu_s_W;     
    
    // Inv Cov    
    const double* J_i_N;
    const double* J_i_S;
    const double* J_i_E;
    const double* J_i_W;    

    const double* J_s_N;
    const double* J_s_S;
    const double* J_s_E;
    const double* J_s_W;

    bool isNvalid = 0;
    bool isSvalid = 0;
    bool isEvalid = 0;
    bool isWvalid = 0;
     
    
    // In the implementation below, if the label of the center pixel is
    // different from the labels of all its (4-conn) nbrs -- that is, it is 
    // a single-pixel superpixel -- then we allow it to die off inspite the fact
    // that this "changes the connectivity" of this superpixel. 

    if (x_greater_than_1){
        N = seg[idx-xdim]; // the label, above
        set_nbrs(idx,xdim,ydim,x_greater_than_1,y_greater_than_1,x_smaller_than_xdim_minus_1,y_smaller_than_ydim_minus_1,seg,nbrs,N);
        isNvalid=ischangbale_by_nbrs(nbrs);   
        if (isNvalid){        
            mu_i_N = mu_i + N * NUM_OF_CHANNELS;
            mu_s_N = mu_s + N * 2;
            if (calculate_cov){
                J_i_N = J_i + N * NUM_OF_CHANNELS * NUM_OF_CHANNELS;
                J_s_N = J_s + N * 4;
            }
        }
        else{        
            N=BAD_TOPOLOGY_LABEL;
            if (N==C) return; // Bug fix, 03/12/2015, Oren Freifeld
        }
        
            
    }
    

       
    if (y_greater_than_1){
        W = seg[idx-1];  // left
        set_nbrs(idx,xdim,ydim,x_greater_than_1,y_greater_than_1,x_smaller_than_xdim_minus_1,y_smaller_than_ydim_minus_1,seg,nbrs,W);
        isWvalid=ischangbale_by_nbrs(nbrs);   
        if (isWvalid){     
            mu_i_W = mu_i + W * NUM_OF_CHANNELS;
            mu_s_W = mu_s + W * 2;
            if (calculate_cov){
                J_i_W = J_i + W * NUM_OF_CHANNELS * NUM_OF_CHANNELS;
                J_s_W = J_s + W * 4;
            }
        }
        else{        
            W=BAD_TOPOLOGY_LABEL;
            if (W==C) return; // Bug fix, 03/12/2015, Oren Freifeld
        }
    }

    if (y_smaller_than_ydim_minus_1){
        S = seg[idx+xdim]; // below
        set_nbrs(idx,xdim,ydim,x_greater_than_1,y_greater_than_1,x_smaller_than_xdim_minus_1,y_smaller_than_ydim_minus_1,seg,nbrs,S);
        isSvalid=ischangbale_by_nbrs(nbrs);   
        if (isSvalid){
            mu_i_S = mu_i + S * NUM_OF_CHANNELS;
            mu_s_S = mu_s + S * 2;
            if (calculate_cov){
                J_i_S = J_i + S * NUM_OF_CHANNELS * NUM_OF_CHANNELS;
                J_s_S = J_s + S * 4;
            }
        }
        else{        
            S=BAD_TOPOLOGY_LABEL;
            if (S==C) return; // Bug fix, 03/12/2015, Oren Freifeld
        }
    }   
    if (x_smaller_than_xdim_minus_1){
        E = seg[idx+1];  // right
        set_nbrs(idx,xdim,ydim,x_greater_than_1,y_greater_than_1,x_smaller_than_xdim_minus_1,y_smaller_than_ydim_minus_1,seg,nbrs,E);
        isEvalid=ischangbale_by_nbrs(nbrs);   
        if (isEvalid){      
            mu_i_E = mu_i + E * NUM_OF_CHANNELS;
            mu_s_E = mu_s + E * 2;
            if (calculate_cov){
                J_i_E = J_i + E * NUM_OF_CHANNELS * NUM_OF_CHANNELS;
                J_s_E = J_s + E * 4;
            }
        }
        else{        
            E=BAD_TOPOLOGY_LABEL;
            if (E==C) return; // Bug fix, 03/12/2015, Oren Freifeld
        }      
    }           


    double pt[2];
    pt[0]=(double)x;
    pt[1]=(double)y;

    

    //---------------
    // log-likelihood  (ignoring constants)
    //---------------   
    double resN = cal_posterior(isNvalid, N, calculate_cov, imgC, pt, log_counts, prior_weight, 
                mu_i_N, mu_s_N, J_i_N, J_s_N, logdet_Sigma_i,logdet_Sigma_s, i_std, s_std, false);
    
    double resS = cal_posterior(isSvalid, S, calculate_cov, imgC, pt, log_counts, prior_weight, 
                mu_i_S, mu_s_S, J_i_S, J_s_S, logdet_Sigma_i,logdet_Sigma_s, i_std, s_std, false);
    
    double resE = cal_posterior(isEvalid, E, calculate_cov, imgC, pt, log_counts, prior_weight, 
                mu_i_E, mu_s_E, J_i_E, J_s_E, logdet_Sigma_i,logdet_Sigma_s, i_std, s_std , false);
    
    double resW = cal_posterior(isWvalid, W, calculate_cov, imgC, pt, log_counts, prior_weight,
                mu_i_W, mu_s_W, J_i_W, J_s_W, logdet_Sigma_i,logdet_Sigma_s, i_std, s_std, false);



    bool all_are_valid = (isNvalid || N==OUT_OF_BOUNDS_LABEL) && 
                         (isSvalid || S==OUT_OF_BOUNDS_LABEL) && 
                         (isEvalid || E==OUT_OF_BOUNDS_LABEL) && 
                         (isWvalid || W==OUT_OF_BOUNDS_LABEL);
    

    if (!all_are_valid)  return;
    
    //double res_max = -1; // some small negative number (use when using l)
    double res_max = log(.000000000000000001); // (use when using ll)
    
    int arg_max = C; // i.e., no change
    
    // In the tests below, the order matters: 
    // E.g., testing (res_max<resN && isNvalid) is wrong!
    // The reason is that if isNvalid, then the test max<resN has no meaning.
    // The correct test is thus isNvalid && res_max<resN. 
    
    
    if (isNvalid && res_max<resN ){ 
        res_max=resN;
        arg_max=N;
    }
    
    if (isSvalid && res_max<resS ){
        res_max=resS;
        arg_max=S;
    }

    if (isEvalid && res_max<resE){
        res_max=resE;
        arg_max=E;
    }

    if (isWvalid && res_max<resW){
        res_max=resW;
        arg_max=W;
    }     

    // update seg
    seg[idx]=arg_max;     
    return;   
}
