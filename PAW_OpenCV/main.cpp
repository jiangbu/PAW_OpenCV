#include <iostream>						// Console I/O
#include <sstream>						// String to number conversion
#include <fstream>

#include <opencv2/core.hpp>				// Basic OpenCV structures
#include <opencv2/core/utility.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/imgproc.hpp>			// Image processing methods for the CPU
#include <opencv2/imgcodecs.hpp>		// Read images
#include <opencv2/highgui.hpp>

#include <opencv2/cudaarithm.hpp>		// CUDA structures and methods
#include <opencv2/cudafilters.hpp>

#include "PAW.hpp"
#include "TIE.hpp"
#include "saveImage.hpp"
#include "dataSave.hpp"

using namespace std;
using namespace cv;

int main()
{	
	//cout << cv::getBuildInformation() << endl;
	Mat image = imread("LetterM.tif", cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
	image.convertTo(image, CV_32F);
	cuda::GpuMat imgRaw;
	imgRaw.upload(image);

	PAW paw;
	
	double t = (double)getTickCount();
	paw.setQuads(imgRaw);
	paw.calcTilt();
	paw.calcPhase();
		
	//normilaze for display
	cuda::multiply(paw.phaseImg, Scalar::all(1000), paw.phaseImg);
	cuda::normalize(paw.phaseImg, paw.phaseImg, 0, 255, NORM_MINMAX, CV_8U);
	namedWindow("phase image", WINDOW_OPENGL);
	imshow("phase image", paw.phaseImg);

	// TIE
	image = imread("chart_Thorlabs.tif", cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
	image.convertTo(image, CV_32F);
	cuda::GpuMat TIERawImg;
	TIERawImg.upload(image);
	
	TIE tie;
	tie.calcIntensity(TIERawImg);
	tie.calcTilt();
	tie.calcPhase();

	for (int i = 0; i < 100; i++)
	{
		tie.calcIntensity(TIERawImg);
		tie.calcTilt();
		tie.calcPhase();
	}

	t = 10 * (double(getTickCount()) - t) / getTickFrequency();
	cout << t << "ms" << endl;

	Mat phaseImage;
	tie.phaseImg.download(phaseImage);
	phaseImage.convertTo(phaseImage, CV_64FC1);
	if (DataSave(phaseImage, ".", "phaseImg", "phaseImg") == EXIT_FAILURE)
		cout << "phase save failed" << endl;

	namedWindow("TIEIntensity", WINDOW_OPENGL);
	namedWindow("TIETilt", WINDOW_OPENGL);
	namedWindow("TIEPhase", WINDOW_OPENGL);
	cuda::normalize(tie.tx, tie.tx, 0, 255, NORM_MINMAX, CV_8UC1);
	cuda::normalize(tie.intensityImg, tie.intensityImg, 0, 255, NORM_MINMAX, CV_8UC1);
	cuda::normalize(tie.phaseImg, tie.phaseImg, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("TIETilt", tie.tx);
	imshow("TIEIntensity", tie.intensityImg);
	imshow("TIEPhase", tie.phaseImg);

	waitKey();
	return 0;
}