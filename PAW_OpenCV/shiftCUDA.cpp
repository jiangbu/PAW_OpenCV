#include "shiftCUDA.hpp"

void shiftCUDA(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::Point2f delta, int fill, cv::Scalar value) 
{
	// error checking
	assert(fabs(delta.x) < src.cols && fabs(delta.y) < src.rows);

	// split the shift into integer and subpixel components
	//cv::Point2i deltai(ceil(delta.x), ceil(delta.y));
	cv::Point2i deltai(round(delta.x), round(delta.y));
	cv::Point2f deltasub(fabs(delta.x - deltai.x), fabs(delta.y - deltai.y));

	// INTEGER SHIFT
	// first create a border around the parts of the Mat that will be exposed
	int t = 0, b = 0, l = 0, r = 0;
	if (deltai.x > 0) l = deltai.x;
	if (deltai.x < 0) r = -deltai.x;
	if (deltai.y > 0) t = deltai.y;
	if (deltai.y < 0) b = -deltai.y;

	cv::cuda::GpuMat padded;
	cv::cuda::copyMakeBorder(src, padded, t, b, l, r, fill, value);

	// construct the region of interest around the new matrix
	cv::Rect roi = cv::Rect(std::max(-deltai.x, 0), std::max(-deltai.y, 0), 0, 0) + src.size();
	dst = padded(roi);
}
