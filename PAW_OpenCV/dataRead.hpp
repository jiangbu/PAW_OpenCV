#ifndef DATASAVE_HPP_
#define DATASAVE_HPP_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <mat.h>
using namespace cv;
using namespace std;
Mat DataRead(string datapath, string filename, string matname);

#endif