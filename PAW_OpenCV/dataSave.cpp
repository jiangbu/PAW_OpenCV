#include "dataSave.hpp"
int DataSave(Mat SrcMat, string datapath, string filename, string matname)
{
	SrcMat = SrcMat.t();
	int rows = SrcMat.rows;
	int cols = SrcMat.cols;
	Mat SaveMat(rows, cols, CV_64FC1);
	if (SrcMat.type() != CV_64FC1)//判断是否为double型的矩阵
	{
		for (int i = 0; i<rows; i++)
		{
			float *src_ptr = SrcMat.ptr<float>(i);
			double *save_ptr = SaveMat.ptr<double>(i);
			for (int j = 0; j<cols; j++)
			{
				*save_ptr++ = src_ptr[j];
			}
		}
	}
	else
		SaveMat = SrcMat.clone();
	MATFile *pmatFile = matOpen((datapath + "\\" + filename + ".mat").c_str(), "w"); //以写的形式打开
	if (pmatFile == NULL)
	{
		cout << "MatOpen error!!!" << endl;
		return EXIT_FAILURE;
	}
	mxArray *pMxArray = mxCreateDoubleMatrix(cols, rows, mxREAL);  //需注意matlab按列储存mat文件，与一般的二维数组、opencv的Mat以行存储不一致，mxCreateDoubleMatrix创建的是double型的pMxArray结构，对应的还有mxCreateNumericMatrix（整型）、mxCreateLogicalMatrix（布尔型）
	if (pMxArray == NULL)
	{
		cout << "Unable to create mxArray, maybe out of memory!!!" << endl;
		return EXIT_FAILURE;
	}
	memcpy((void *)(mxGetPr(pMxArray)), (void *)SaveMat.ptr<float>(0), sizeof(double)*rows*cols);//内存copy，注意源和目的地址的数据类型需一致
	cout << "Saved data----" << datapath << "\\" << filename << endl;
	int status = matPutVariable(pmatFile, matname.c_str(), pMxArray);//put到文件
	if (status != 0) {
		cout << "Error using matPutVariable!!!" << endl;
		return EXIT_FAILURE;
	}
	mxDestroyArray(pMxArray);//结构销毁
	if (matClose(pmatFile) != 0) {
		cout << "Error closing file " << pmatFile << endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}