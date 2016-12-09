#include "convCUDA2.hpp"
#include <vector>
#include <iostream>

using namespace std;

cuda::GpuMat convCUDA2(cuda::GpuMat A, cuda::GpuMat B)
{
	cuda::GpuMat dftA, dftB;
	A.convertTo(dftA, CV_32FC1);
	B.convertTo(dftB, CV_32FC1);

	int r = A.size().height;
	int c = A.size().width;

	vector<cuda::GpuMat> planeA, planeB;

	cuda::GpuMat zero(r, c, CV_32FC1);
	zero.setTo(Scalar::all(0));

	planeA.push_back(dftA);
	planeA.push_back(zero);
	planeB.push_back(dftB);
	planeB.push_back(zero);
	merge(planeA, dftA);
	merge(planeB, dftB);
	cuda::dft(dftA, dftA, A.size());
	cuda::dft(dftB, dftB, B.size());

	cuda::GpuMat conv;
	cuda::mulSpectrums(dftA, dftB, conv, 0);
	cuda::dft(conv, conv, A.size(), DFT_SCALE + DFT_INVERSE);
	split(conv, planeA);
	//return planeA[0](Rect(0, 0, A.size().width, A.size().height));
	return planeA[0];
}