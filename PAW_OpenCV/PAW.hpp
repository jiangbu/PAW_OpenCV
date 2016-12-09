#ifndef PAW_HPP_
#define PAW_HPP_

#include <opencv2/core.hpp>      // Basic OpenCV structures
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <vector>

using namespace std;
using namespace cv;

class PAW
{
public:
	PAW();
	void setQuads(cuda::GpuMat imgRaw);
	void calcTilt();	
	void calcPhase();
	
	//~PAW();

	Mat tx_n, ty_n, phaseImg_n;
	cuda::GpuMat tx, ty, phaseImg;
	cuda::GpuMat sumI, I1, I2, I3, I4, imgBlank, imgDark;
private:
	void calcFourierFilter();
	
	cuda::GpuMat fourier_filter; //fourier integration filter
	Size padSize;	//size of padded matrix for FFT
	Size s;		//size of a quad matrix
	vector<vector<float>> quadRect; //quads rect values
	vector<vector<float>> quadOffset; //quads shift values
	cuda::GpuMat tmp1, tmp2, tmp3, tmp4;	//pre allocated Mat for speeding up
	double lambda, NAi, M_tot, pixel_size;
};

#endif