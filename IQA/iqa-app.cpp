#include "version.h"
#include "IQA.h"


CIqa g_iqa;

int main(int argc, char *argv[])
{
	if (argc != 2 && argc!=3)
	{
		cout << "err: no argv[1]";
		return -1;
	}
	string str(argv[1]);
	Mat a;
	float mos;
	a = imread(str);
	if (a.empty())
	{
		cout << "err: input file not exist";
		return -2;
	}

	int bHaveBI = 1;
	if (argc == 3)
		bHaveBI = atoi(argv[2]);
	//cout << str << ",";
	cout << g_iqa.calcMos(a, bHaveBI);
	//g_iqa.printGradIndex(a);
	//g_iqa.printBlockingIndex(a);
	//g_iqa.printHSVBlurIndex(a);
	//g_iqa.printIqa(a);
	//g_iqa.printNRSS(a);
	//Mat b;
	//blur(a, b, Size(3, 3));
	//imwrite(argv[2], b);

	return 0;
}
