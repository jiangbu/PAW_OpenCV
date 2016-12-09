#ifndef DATASAVE_HPP_
#define DATASVAE_HPP_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <mat.h>
using namespace cv;
using namespace std;
int DataSave(Mat SrcMat, string datapath, string filename, string matname);

#endif