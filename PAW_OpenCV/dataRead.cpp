#include "dataRead.hpp"

Mat DataRead(string datapath, string filename, string matname)
{
	MATFile *pmatFile = matOpen((datapath + "\\" + filename + ".mat").c_str(), "r");
	if (pmatFile == NULL)
		cout << "MatOpen error!!!" << endl;
	mxArray *pMxArray = matGetVariable(pmatFile, matname.c_str()); //从文件中获取数据到mxArray结构中
	if (pMxArray == NULL)
		cout << "Error reading existing matrix " << matname << "!!!" << endl;
	double *ReadArray = (double*)mxGetData(pMxArray);
	int cols = mxGetM(pMxArray);//行列存储方式不一致，需注意
	int rows = mxGetN(pMxArray);
	Mat ReadMat(rows, cols, CV_32FC1);  //此处主要是自己工程其他模块需使用float型的
	for (int i = 0; i<rows; i++)
	{
		for (int j = 0; j<cols; j++)
		{
			ReadMat.at<float>(i, j) = (float)ReadArray[i*cols + j];
		}
	}
	mxDestroyArray(pMxArray);
	if (matClose(pmatFile) != 0)
		cout << "Error closing file " << pmatFile << endl;
	cout << filename << " read success!!!" << endl;
	return ReadMat.t();
}