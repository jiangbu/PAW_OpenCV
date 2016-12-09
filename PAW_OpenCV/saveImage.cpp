#include "saveImage.hpp"

void saveImage(cv::Mat src, string filename)
{
	ofstream fout(filename);
	if (!fout)
	{
		cout << "File Not Opened" << endl;
		return;
	}
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			fout << src.at<float>(i, j) << "\t";
		}
		fout << endl;
	}
	fout.close();
	cout << filename << " saved successfully." << endl;
}