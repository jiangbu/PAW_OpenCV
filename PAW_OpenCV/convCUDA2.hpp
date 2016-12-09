#ifndef CONVCUDA2_HPP_
#define CONVCUDA2_HPP_

#include <opencv2/cudaarithm.hpp>		// CUDA structures and methods
#include <opencv2/cudafilters.hpp>

using namespace cv;
/*! @brief
calculate convulation of two matrices A and B,  thye must have the same size
*/

cuda::GpuMat convCUDA2(cuda::GpuMat A, cuda::GpuMat B);


#endif