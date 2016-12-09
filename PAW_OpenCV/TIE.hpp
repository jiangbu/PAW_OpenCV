#ifndef TIE_HPP_
#define TIE_HPP_

#include <opencv2/core.hpp>      // Basic OpenCV structures
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <vector>

using namespace cv;
using namespace std;

class TIE
{
public:
	TIE();
	void calcIntensity(cuda::GpuMat imgRaw);
	void calcTilt();
	void calcPhase();
	
	cuda::GpuMat tx, ty, intensityImg, phaseImg;
	cuda::GpuMat imgBlank, imgDark;
private:
	void calcFourierFilter();
	void calcIllumination();
	
	cuda::GpuMat fi;     // illumination profile
	cuda::GpuMat fi_x;
	cuda::GpuMat fi_y;
	Mat tform, tformTx, tformTy;

	cuda::GpuMat fourier_filter; //fourier integration filter

	Size padSize;	//size of padded matrix for FFT
	Size S;		//size of a quad matrix
	Mat RectChart; //crop rect values
	double resizeScale;
	cuda::GpuMat tmp1, tmp2, tmp3, tmp4;	//pre allocated Mat for speeding up
	double lambda, NAi, M_tot, pixel_size, Z;
};

#endif