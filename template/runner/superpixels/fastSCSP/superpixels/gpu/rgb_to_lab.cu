/*
* Author:
* Yixin Li, Email: liyixin@mit.edu
* convert the image from RGB to LAB
*/
__global__ void rgb_to_lab( double * img, const int nPts) {

	// getting the index of the pixel
	const int t = threadIdx.x + blockIdx.x * blockDim.x; 
	if (t>=nPts) return;

	double sR = img[3*t];
	double sG = img[3*t+1];
	double sB = img[3*t+2];

	if (sR!=sR || sG!=sG || sB!=sB) return;

	//RGB (D65 illuninant assumption) to XYZ conversion
	double R = sR/255.0;
	double G = sG/255.0;
	double B = sB/255.0;

	double r, g, b;
	if(R <= 0.04045)	r = R/12.92;
	else				r = pow((R+0.055)/1.055,2.4);
	if(G <= 0.04045)	g = G/12.92;
	else				g = pow((G+0.055)/1.055,2.4);
	if(B <= 0.04045)	b = B/12.92;
	else				b = pow((B+0.055)/1.055,2.4);

	double X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	double Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	double Z = r*0.0193339 + g*0.1191920 + b*0.9503041;

	
	//convert from XYZ to LAB

	double epsilon = 0.008856;	//actual CIE standard
	double kappa   = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X/Xr;
	double yr = Y/Yr;
	double zr = Z/Zr;

	double fx, fy, fz;
	if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
	else				fx = (kappa*xr + 16.0)/116.0;
	if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
	else				fy = (kappa*yr + 16.0)/116.0;
	if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
	else				fz = (kappa*zr + 16.0)/116.0;

	double lval = 116.0*fy-16.0;
	double aval = 500.0*(fx-fy);
	double bval = 200.0*(fy-fz);

	img[3*t] = lval;
	img[3*t+1] = aval;
	img[3*t+2] = bval;
}