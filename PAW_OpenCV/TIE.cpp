#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/opengl.hpp>

#include "TIE.hpp"
#include "convCUDA2.hpp"
#include "readRegVal.hpp"
#include "shift.hpp"
#include "saveImage.hpp"
#include "dataRead.hpp"

using namespace std;

TIE::TIE()
{
	lambda = 0.6;
	M_tot = 6;
	NAi = 0.2;
	pixel_size = 8;
	Z = 350;
		
	//Rect
	RectChart = DataRead("D:\\Copy\\01 Matlab Projects\\PAW Code v3.1\\calib_files", "camera2_register", "rect_chart_two");
		
	//resize scale
	Mat matRead;
	matRead = DataRead("D:\\Copy\\01 Matlab Projects\\PAW Code v3.1\\calib_files", "camera2_register", "resize_scale");
	resizeScale = (matRead.at<float>(0, 0));
	
	//dark image
	Mat img1 = imread("dark_Thorlabs.tif", cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
	if (img1.empty())
		cout << "TIE dark image read failed" << endl;
	img1.convertTo(img1, CV_32F);
	imgDark.upload(img1);

	//blank image
	Mat img2 = imread("blank_Thorlabs.tif", cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
	if (img2.empty())
		cout << "TIE blank image read failed" << endl;
	img2.convertTo(img2, CV_32F);
	//subtract(img2, img1, img2);
	//max(img2, 1, img2);
	imgBlank.upload(img2);

	S.width = RectChart.at<float>(0, 2);
	S.height = RectChart.at<float>(0, 3);
	padSize.height = getOptimalDFTSize(S.height);
	padSize.width = getOptimalDFTSize(S.width);
	
	calcIllumination();
	calcFourierFilter();

	//tform
	tform = DataRead("D:\\Copy\\01 Matlab Projects\\PAW Code v3.1\\calib_files", "camera2_register", "tform_T");
	tformTx = DataRead("D:\\Copy\\01 Matlab Projects\\PAW Code v3.1\\calib_files", "camera2_register", "tform_tx_T");
	tformTy = DataRead("D:\\Copy\\01 Matlab Projects\\PAW Code v3.1\\calib_files", "camera2_register", "tform_ty_T");
}

void TIE::calcIllumination()
{
	Mat kernalX = Mat(padSize, CV_32F);
	Mat kernalY = Mat(padSize, CV_32F);
	Mat kernal = Mat(padSize, CV_32F);

	int c = padSize.width;
	int r = padSize.height;

	double w = (Z * NAi) * (Z * NAi);

	for (double i = 0; i < c; i++)
		kernalX.col(i) = i - floor(c / 2);
	for (double j = 0; j < r; j++)
		kernalY.row(j) = j - floor(r / 2);

	kernal = -kernalX.mul(kernalX) - kernalY.mul(kernalY);
	kernal = kernal / w;
	exp(kernal, kernal);
	divide(kernal, sum(kernal), kernal);

	Mat fx = kernal.mul(kernalX) / Z;
	Mat fy = kernal.mul(kernalY) / Z;

	//saveImage(kernal, "kernal.txt");

	shift(kernal, kernal, Point(ceil(padSize.width / 2), ceil(padSize.height / 2)), BORDER_WRAP);
	shift(fx, fx, Point(ceil(padSize.width / 2), ceil(padSize.height / 2)), BORDER_WRAP);
	shift(fy, fy, Point(ceil(padSize.width / 2), ceil(padSize.height / 2)), BORDER_WRAP);

	fi.upload(kernal);
	fi_x.upload(fx);
	fi_y.upload(fy);
}

void TIE::calcIntensity(cuda::GpuMat imgRaw)
{
	cuda::GpuMat img;
	cuda::divide(imgRaw, imgBlank, img);
	cuda::resize(img, img, Size(), resizeScale, resizeScale, INTER_CUBIC);
	img = img(Rect(RectChart.at<float>(0, 0), RectChart.at<float>(0, 1), RectChart.at<float>(0, 2), RectChart.at<float>(0, 3)));
	cuda::warpAffine(img, intensityImg, tform, S);	
	cuda::copyMakeBorder(intensityImg, intensityImg, 0, padSize.height - S.height, 0, padSize.width - S.width, BORDER_REPLICATE);	
}

void TIE::calcTilt()
{
	cuda::GpuMat I1;
	I1 = convCUDA2(intensityImg, fi);

	tx = convCUDA2(intensityImg, fi_x);
	cuda::divide(tx, I1, tx);
	tx = tx(Rect(0, 0, S.width, S.height));
	Scalar meanTilt, sumTilt;
	sumTilt = cuda::sum(tx);
	meanTilt = sumTilt / (S.width*S.height);
	cuda::subtract(tx, meanTilt, tx);

	ty = convCUDA2(intensityImg, fi_y);
	cuda::divide(ty, I1, ty);
	ty = ty(Rect(0, 0, S.width, S.height));
	sumTilt = cuda::sum(ty);
	meanTilt = sumTilt / (S.width*S.height);
	cuda::subtract(ty, meanTilt, ty);
}

void TIE::calcFourierFilter()
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

void TIE::calcPhase()
{
	cuda::GpuMat padTx, padTy;          //expand input image to optimal size
	int m = padSize.height;
	int n = padSize.width;
	//cout << m - tx.rows << " " << n - tx.cols << endl;
	cuda::copyMakeBorder(tx, padTx, 0, m - tx.rows, 0, n - tx.cols, BORDER_CONSTANT, Scalar::all(0));
	cuda::copyMakeBorder(ty, padTy, 0, m - tx.rows, 0, n - tx.cols, BORDER_CONSTANT, Scalar::all(0));

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
	phaseImg = G_xy(Rect(0, 0, S.width, S.height));
	cuda::split(phaseImg, planes);
	phaseImg = planes[1];
}