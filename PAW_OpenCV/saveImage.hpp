#ifndef SAVEIMAGE_HPP_
#define SAVEIMAGE_HPP_
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2\core.hpp>

/*! @brief save the Mat image to a txt file. Must be float type.
*/

using namespace std;

void saveImage(cv::Mat src, string filename);

#endif