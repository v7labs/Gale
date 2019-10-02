/*
* Authors:
* Oren Freifeld, freifeld@csail.mit.edu
* Yixin Li, Email: liyixin@mit.edu
*/
__global__ void clear_fields(int * count, int * non_NaN_count, int * NaN_count, double * log_count,
	 int * mu_i_h, int * mu_s_h, double * mu_i, double * mu_s, double * sigma_s_h,
	 const int dim_i, const int nsuperpixel){

	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nsuperpixel) return;

	count[k] = 0;
	non_NaN_count[k] = 0;
	NaN_count[k] = 0;
	log_count[k] = 0.0;

#pragma unroll 
	for (int d = 0; d<dim_i; d=d+1){
		mu_i_h[dim_i*k+d] = 0;
		mu_i[dim_i*k+d] = 0.0;
	}

	mu_s_h[2*k] = mu_s_h[2*k+1] = 0;
	mu_s[2*k] =  mu_s[2*k+1] = 0.0;

    sigma_s_h[3*k] = sigma_s_h[3*k+1] = sigma_s_h[3*k+2] = 0;
}



__global__ void sum_by_label(
	double * img, bool * img_isNaN, int * seg, 
	int * count, int * non_NaN_count, int * NaN_count, int * mu_i_h, int * mu_s_h, unsigned long long int * sigma_s_h, 
	const int xdim, const int ydim, const int dim_i, const int nPts) {

	// getting the index of the pixel
	const int t = threadIdx.x + blockIdx.x * blockDim.x; 
	if (t>=nPts) return;

	//get the label: superpixel t has label k
	const int k = seg[t];

	atomicAdd(&count[k] , 1);

	//If the pixel value is not NaN
	// increment non_NaN_count and mu_i of the corresponding superpixel
	if (img_isNaN[t]){
		atomicAdd(&NaN_count[k], 1);
	}else{
		atomicAdd(&non_NaN_count[k], 1);
#pragma unroll 
		for (int d = 0; d<dim_i; d=d+1){
			atomicAdd(&mu_i_h[dim_i*k+d], img[dim_i*t+d]);
		}

	}


	int x =  t % xdim;
	int y = t / xdim; 
	int xx = x * x;
	int xy = x * y;
	int yy = y * y;
	atomicAdd(&mu_s_h[2*k], x);
	atomicAdd(&mu_s_h[2*k+1], y);
    atomicAdd(&sigma_s_h[3*k], xx);
	atomicAdd(&sigma_s_h[3*k+1], xy);
	atomicAdd(&sigma_s_h[3*k+2], yy);
}


__global__ void calculate_mu_and_sigma(
	int * counts, int * non_NaN_counts, double* log_count, int * mu_i_h, int * mu_s_h, 
	double * mu_i, double * mu_s, 
	unsigned long long int * sigma_s_h, double * prior_sigma_s,
	double * sigma_s, double * logdet_Sigma_s, double * J_s,
	const int prior_count, const int dim_i, const int nsuperpixel)
{
	const int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel) return;

	double count = double (counts[k]);
	double non_NaN_count =	double (non_NaN_counts[k]);
	double mu_x = 0.0;
	double mu_y = 0.0;

	//calculate the mean
	if (counts[k]>0){
		log_count[k] = log(count);
	    mu_x = mu_s_h[2*k] / count;   
	    mu_y = mu_s_h[2*k+1]/ count;  
		mu_s[2*k] =  mu_x; 
	    mu_s[2*k+1] = mu_y;
    }

    if(non_NaN_counts[k]>0){
#pragma unroll 
	    for (int d = 0; d<dim_i; d=d+1){
	    	 mu_i[dim_i*k+d] = mu_i_h[dim_i*k+d]/non_NaN_count;
		}
	}

	//calculate the covariance
	double C00,C01,C11; 
	C00 = C01 = C11  = 0;
	double total_count = double(counts[k] + prior_count); 
	if (count > 3){	    
	    //update cumulative count and covariance
	    C00= sigma_s_h[3*k] - mu_x * mu_x * count;
	    C01= sigma_s_h[3*k+1] - mu_x * mu_y * count;
	    C11= sigma_s_h[3*k+2] - mu_y * mu_y * count;
	    total_count = total_count -3;
	}

    C00 =  (prior_sigma_s[k*4] + C00) / total_count;
    C01 =  (prior_sigma_s[k*4+1] + C01)/ total_count;
    C11 =  (prior_sigma_s[k*4+3] + C11) / total_count;
	
    double detC = C00 * C11 - C01 * C01;
    if (detC <= 0){
        C00 = C00 + 0.00001;
        C11 = C11 + 0.00001;
        detC = C00*C11-C01*C01;   
        if(detC <=0) detC = 0.0001;//hack
    }        
    
    //set the sigma_space
    sigma_s[k*4] =  C00;
    sigma_s[k*4+1] =  C01;
    sigma_s[k*4+2] =  C01;
    sigma_s[k*4+3] =  C11;

    //Take the inverse of sigma_space to get J_space
    J_s[k*4] = C11 / detC;     
    J_s[k*4+1] = -C01/ detC; 
    J_s[k*4+2] = -C01/ detC;
    J_s[k*4+3] = C00/ detC; 
    
    logdet_Sigma_s[k] = log(detC);
}







__global__ void clear_fields_2(int * count, int * non_NaN_count,
	int * mu_i_h, int * mu_s_h,
	const int dim_i, const int nsuperpixel){
	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label
	if (k>=nsuperpixel) return;

	//clear the fields

	count[k] = 0;
	non_NaN_count[k] = 0;

#pragma unroll 
	for (int d = 0; d<dim_i; d=d+1){
		mu_i_h[dim_i*k+d] = 0;
	}

	mu_s_h[2*k] = mu_s_h[2*k+1] = 0;
}


__global__ void sum_by_label_2(double * img, bool * img_isNaN, int * seg, 
	int * count, int * non_NaN_count, 
	int * mu_i_h, int * mu_s_h,
	const int xdim, const int ydim, const int dim_i, const int nPts) {

	// getting the index of the pixel
	const int t = threadIdx.x + blockIdx.x *blockDim.x; 
	if (t>=nPts) return;

	//get the label
	const int k = seg[t];
	atomicAdd(&count[k] , 1);

	atomicAdd(&mu_s_h[2*k], t % xdim);
	atomicAdd(&mu_s_h[2*k+1], t / xdim);

	//If the pixel value is not NaN
	if (!img_isNaN[t]){
		atomicAdd(&non_NaN_count[k] , 1);
#pragma unroll 
		for (int d = 0; d<dim_i; d=d+1){
			atomicAdd(&mu_i_h[dim_i*k+d], img[dim_i*t+d]);
		}

	}
}


__global__ void calculate_mu(
	int * counts, int * non_NaN_counts, int * mu_i_h, int * mu_s_h, double * mu_i, double * mu_s,
	const int dim_i, const int nsuperpixel)
{
	const int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nsuperpixel) return;
	double count = double (counts[k]);
	double non_NaN_count =	double (non_NaN_counts[k]);

	if (count>0){
		mu_s[2*k] =  mu_s_h[2*k] / count; 
	    mu_s[2*k+1] = mu_s_h[2*k+1]/ count; 
#pragma unroll 
	    for (int d = 0; d<dim_i; d=d+1){
	    	 mu_i[dim_i*k+d] = mu_i_h[dim_i*k+d]/non_NaN_count;
		}
	}
}