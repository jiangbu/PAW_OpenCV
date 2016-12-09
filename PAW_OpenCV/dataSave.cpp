#include "dataSave.hpp"
int DataSave(Mat SrcMat, string datapath, string filename, string matname)
{
	SrcMat = SrcMat.t();
	int rows = SrcMat.rows;
	int cols = SrcMat.cols;
	Mat SaveMat(rows, cols, CV_64FC1);
	if (SrcMat.type() != CV_64FC1)//�ж��Ƿ�Ϊdouble�͵ľ���
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
	MATFile *pmatFile = matOpen((datapath + "\\" + filename + ".mat").c_str(), "w"); //��д����ʽ��
	if (pmatFile == NULL)
	{
		cout << "MatOpen error!!!" << endl;
		return EXIT_FAILURE;
	}
	mxArray *pMxArray = mxCreateDoubleMatrix(cols, rows, mxREAL);  //��ע��matlab���д���mat�ļ�����һ��Ķ�ά���顢opencv��Mat���д洢��һ�£�mxCreateDoubleMatrix��������double�͵�pMxArray�ṹ����Ӧ�Ļ���mxCreateNumericMatrix�����ͣ���mxCreateLogicalMatrix�������ͣ�
	if (pMxArray == NULL)
	{
		cout << "Unable to create mxArray, maybe out of memory!!!" << endl;
		return EXIT_FAILURE;
	}
	memcpy((void *)(mxGetPr(pMxArray)), (void *)SaveMat.ptr<float>(0), sizeof(double)*rows*cols);//�ڴ�copy��ע��Դ��Ŀ�ĵ�ַ������������һ��
	cout << "Saved data----" << datapath << "\\" << filename << endl;
	int status = matPutVariable(pmatFile, matname.c_str(), pMxArray);//put���ļ�
	if (status != 0) {
		cout << "Error using matPutVariable!!!" << endl;
		return EXIT_FAILURE;
	}
	mxDestroyArray(pMxArray);//�ṹ����
	if (matClose(pmatFile) != 0) {
		cout << "Error closing file " << pmatFile << endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}