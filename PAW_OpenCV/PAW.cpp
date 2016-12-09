#include <vector>
#include <fstream>
#include <opencv2\highgui\highgui.hpp>

#include "PAW.hpp"
#include "readRegVal.hpp"
#include "shift.hpp"
#include "shiftCUDA.hpp"

using namespace cv;
using namespace std;

PAW::PAW()
{
	if (readRegVal(quadRect, "cropVals.txt") && readRegVal(quadOffset, "offset_sub.txt"))
	{
		
		s.width = quadRect[0][2];
		s.height = quadRect[0][3];
		sumI.create(s, CV_32F);
		I1.create(s, CV_32F);
		I2.create(s, CV_32F);
		I3.create(s, CV_32F);
		I4.create(s, CV_32F);
		tmp1.create(s, CV_32F);
		tmp2.create(s, CV_32F);
		tmp3.create(s, CV_32F);
		tmp4.create(s, CV_32F);

		padSize.height = getOptimalDFTSize(s.height + 100);
		padSize.width = getOptimalDFTSize(s.width + 100);
		
		calcFourierFilter();
	}

	lambda = 0.6;
	M_tot = 20;
	NAi = 0.6;
	pixel_size = 8;

	// read dark image
	Mat img1 = imread("dark.tif", cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
	if (img1.empty())
		cout << "dark image read failed" << endl;
	img1.convertTo(img1, CV_32F);
	imgDark.upload(img1);

	// read blank image
	Mat img2 = imread("blank.tif", cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
	if (img2.empty())
		cout << "blank image read failed" << endl;
	img2.convertTo(img2, CV_32F);
	//subtract(img2, img1, img2);
	//max(img2, 1, img2);
	imgBlank.upload(img2);
	
}

void PAW::setQuads(cuda::GpuMat imgRaw)
{
	//cuda::subtract(imgRaw, imgDark, imgRaw);
	//cuda::max(imgRaw, 1.0, imgRaw);
	cuda::divide(imgRaw, imgBlank, imgRaw);
	/*
	rect:
	|3|4|
	|2|1|
	quads:
	|3|2|
	|4|1|	
	*/
	// swap I2 and I4
	I1 = imgRaw(Rect(quadRect[0][0], quadRect[0][1], quadRect[0][2], quadRect[0][3]));
	I2 = imgRaw(Rect(quadRect[3][0], quadRect[3][1], quadRect[3][2], quadRect[3][3]));
	I3 = imgRaw(Rect(quadRect[2][0], quadRect[2][1], quadRect[2][2], quadRect[2][3]));
	I4 = imgRaw(Rect(quadRect[1][0], quadRect[1][1], quadRect[1][2], quadRect[1][3]));
	
	shiftCUDA(I1, I1, Point2f(quadOffset[1][0], quadOffset[0][0]), BORDER_REPLICATE);
	shiftCUDA(I2, I2, Point2f(quadOffset[1][3], quadOffset[0][3]), BORDER_REPLICATE);
	shiftCUDA(I3, I3, Point2f(quadOffset[1][2], quadOffset[0][2]), BORDER_REPLICATE);
	shiftCUDA(I4, I4, Point2f(quadOffset[1][1], quadOffset[0][1]), BORDER_REPLICATE);
		
	cuda::add(I1, I2, sumI);
	cuda::add(I3, sumI, sumI);
	cuda::add(I4, sumI, sumI);
}

void PAW::calcTilt()
{
	// tx = ((I1 + I2) - (I3 + I4)/ sum
	cuda::add(I1, I2, tx);
	cuda::subtract(tx, I3, tx);
	cuda::subtract(tx, I4, tx);
	cuda::divide(tx, sumI, tx);
	cuda::multiply(tx, -NAi, tx);
	Scalar meanTilt, sumTilt;
	sumTilt = cuda::sum(tx);
	meanTilt = sumTilt / (s.width*s.height);
	cuda::subtract(tx, meanTilt, tx);

	// ty = ((I1 + I4) - (I2 + I3)) / sum
	cuda::add(I1, I4, ty);
	cuda::subtract(ty, I2, ty);
	cuda::subtract(ty, I3, ty);
	cuda::divide(ty, sumI, ty);
	cuda::multiply(ty, -NAi, ty);
	sumTilt = cuda::sum(ty);
	meanTilt = sumTilt / (s.width*s.height);
	cuda::subtract(ty, meanTilt, ty);
}

void PAW::calcFourierFilter()
{
	int r = padSize.height;
	int c = padSize.width;

	Mat k[] = { Mat::zeros(padSize, CV_32F), Mat::zeros(padSize, CV_32F) };

	for (float i = 0; i < c; i++)
		k[0].col(i) = (i - floor(r / 2)) / c;
	for (float j = 0; j < r; j++)
		k[1].row(j) = (j - floor(c / 2)) / r;
	
	Mat kc, knorm, mask;
	merge(k, 2, kc);
	magnitude(k[0], k[1], knorm);
	compare(knorm, 0, mask, CMP_EQ);
	
	// spf = 1./(K[0] + i k[1]) = (k[0] - i k[1])./(k[0]^2 + k[1]^2)
	Mat spf2[] = { Mat::zeros(padSize, CV_32F), Mat::zeros(padSize, CV_32F) };
	Mat D0 = k[0].mul(k[0]) + k[1].mul(k[1]);
	spf2[0] = k[0] / D0;
	spf2[1] = -k[1] / D0;

	//enforce hermitian symmetry
	if ((r % 2) == 0)
	{
		spf2[0].row(0) = 0;
		spf2[1].row(0) = 0;
	}
	if ((c % 2) == 0)
	{
		spf2[0].col(0) = 0;
		spf2[1].col(0) = 0;
	}
	Mat spf;
	merge(spf2, 2, spf);
	cv::divide(spf, 2 * 3.14159265, spf);
	
	//ifftshift
	shift(spf, spf, Point(ceil(spf.cols / 2), ceil(spf.rows / 2)), BORDER_WRAP);
	//accounting for singularity
	spf.at<float>(0, 0) = 0; 
	fourier_filter.upload(spf);	
}

void PAW::calcPhase()
{
	cuda::GpuMat padTx, padTy;          //expand input image to optimal size
	int m = padSize.height;
	int n = padSize.width;
	//cout << m - tx.rows << " " << n - tx.cols << endl;
	cuda::copyMakeBorder(tx, padTx, 50, m - tx.rows-50, 50, n - tx.cols-50, BORDER_CONSTANT, Scalar::all(0));
	cuda::copyMakeBorder(ty, padTy, 50, m - tx.rows - 50, 50, n - tx.cols - 50, BORDER_CONSTANT, Scalar::all(0));
	
	std::vector<cuda::GpuMat> planes;
	planes.push_back(padTx);
	planes.push_back(padTy);
	
	cuda::GpuMat G_xy;
	cuda::merge(planes, G_xy);
	cuda::multiply(G_xy, Scalar::all(2 * 3.14159265 / lambda), G_xy);

	cuda::dft(G_xy, G_xy, padSize, DFT_SCALE);
	//cuda::dft(G_xy, G_xy, padSize);
	cuda::mulSpectrums(G_xy, fourier_filter, G_xy, 0);
	cuda::multiply(G_xy, Scalar::all(pixel_size / M_tot), G_xy);
	
	cuda::dft(G_xy, G_xy, padSize, DFT_INVERSE);
	phaseImg = G_xy(Rect(50,50,s.width,s.height));
	cuda::split(phaseImg, planes);
	phaseImg = planes[1];
}