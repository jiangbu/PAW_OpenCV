#include "readRegVal.hpp"
using namespace std;

int readRegVal(vector<vector<float>>& quadRect, string fileName)
{
	ifstream fstrm;
	string line;
	
	fstrm.open(fileName);
	
	if (fstrm.is_open())
	{
		while (getline(fstrm, line))
		{
			vector<float> lineData;
			stringstream lineStream(line);
			float value;
			// Read an integer at a time from the line
			while (lineStream >> value)
			{
				// Add the integers from a line to a 1D array (vector)
				//value = round(value);
				lineData.push_back(value);
			}
			// When all the integers have been read add the 1D array
			// into a 2D array (as one line in the 2D array)
			quadRect.push_back(lineData);
		}
	}
	else
	{
		cout << "register file open failed" << endl;
		return -1;
	}	
	return 1;
}