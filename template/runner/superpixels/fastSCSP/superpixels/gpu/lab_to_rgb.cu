/*
* Author:
* Yixin Li, Email: liyixin@mit.edu
* convert the image from LAB to RGB
*/
__global__ void lab_to_rgb( double * img, const int nPts) {

	// getting the index of the pixel
	const int t = threadIdx.x + blockIdx.x * blockDim.x; 
	if (t>=nPts) return;
    
    double L = img[3*t];
	double La = img[3*t+1];
	double Lb = img[3*t+2];

	if (L!=L || La!=La || Lb!=Lb) return;

    //convert from LAB to XYZ
    double fy = (L+16) / 116;
	double fx = La/500 + fy;
	double fz = fy-Lb/200;

	double x,y,z;
	double xcube = pow(fx,3); 
	double ycube = pow(fy,3); 
	double zcube = pow(fz,3); 
	if (ycube>0.008856)	y = ycube;
	else				y = (fy-16.0/116.0)/7.787;
	if (xcube>0.008856)	x = xcube;
	else				x = (fx - 16.0/116.0)/7.787;
	if (zcube>0.008856)	z = zcube;
	else				z = (fz - 16.0/116.0)/7.787;

	double X = 0.950456 * x;
	double Y = 1.000 * y;
	double Z = 1.088754 * z;

	//convert from XYZ to rgb
	double R = X *  3.2406 + Y * -1.5372 + Z * -0.4986;
	double G = X * -0.9689 + Y *  1.8758 + Z *  0.0415;
	double B = X *  0.0557 + Y * -0.2040 + Z *  1.0570;

	double r,g,b;
	if (R>0.0031308) r = 1.055 * (pow(R,(1.0/2.4))) - 0.055;
	else             r = 12.92 * R;
	if (G>0.0031308) g = 1.055 * ( pow(G,(1.0/2.4))) - 0.055;
	else             g= 12.92 * G;
	if (B>0.0031308) b = 1.055 * (pow(B, (1.0/2.4))) - 0.055;
	else             b = 12.92 * B;

	img[3*t] = min(255.0, r * 255.0);
	img[3*t+1] = min(255.0, g * 255.0);
	img[3*t+2] = min(255.0, b * 255.0);
}