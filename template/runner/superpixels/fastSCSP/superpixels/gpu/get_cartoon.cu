/*
* Author:
* Yixin Li, Email: liyixin@mit.edu
* Set each pixel value to be the mean of the superpixel it belongs
*/
__global__ void get_cartoon(int* seg, double* mu_i, int* img, const int nChannels, const int nPts) {
	// getting the index of the pixel
	const int t = threadIdx.x + blockIdx.x * blockDim.x; 
	if (t>=nPts) return;

	const int k =  seg[t];
	if (nChannels == 1){
		img[3*t] = img[3*t+1] = img[3*t+2] = max(0.0,mu_i[k]);
	}else{
			img[3*t] = mu_i[3*k];
			img[3*t+1] = mu_i[3*k+1];
			img[3*t+2] = mu_i[3*k+2];
	}
}