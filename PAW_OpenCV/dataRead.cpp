#include "dataRead.hpp"

Mat DataRead(string datapath, string filename, string matname)
{
	MATFile *pmatFile = matOpen((datapath + "\\" + filename + ".mat").c_str(), "r");
	if (pmatFile == NULL)
		cout << "MatOpen error!!!" << endl;
	mxArray *pMxArray = matGetVariable(pmatFile, matname.c_str()); //���ļ��л�ȡ���ݵ�mxArray�ṹ��
	if (pMxArray == NULL)
		cout << "Error reading existing matrix " << matname << "!!!" << endl;
	double *ReadArray = (double*)mxGetData(pMxArray);
	int cols = mxGetM(pMxArray);//���д洢��ʽ��һ�£���ע��
	int rows = mxGetN(pMxArray);
	Mat ReadMat(rows, cols, CV_32FC1);  //�˴���Ҫ���Լ���������ģ����ʹ��float�͵�
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