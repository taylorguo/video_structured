#ifndef CalcHistogram_H
#define CalcHistogram_H

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


class Histogram1D
{
private:
	int histSize[1]; // �������
	float hranges[2]; // ͳ�����ص����ֵ����Сֵ
	const float* ranges[1];
	int channels[1]; // ������һ��ͨ��

public:
	Histogram1D()
	{
		// ׼��1Dֱ��ͼ�Ĳ���
		histSize[0] = 64;
		hranges[0] = 0.0f;
		hranges[1] = 256.0f;
		ranges[0] = hranges;
		channels[0] = 0;
	}

	MatND getHistogram(const Mat &image)
	{
		MatND hist;
		// ����ֱ��ͼ
		calcHist(&image,// Ҫ����ͼ���
			1,                // ֻ����һ��ͼ���ֱ��ͼ
			channels,        // ͨ������
			Mat(),            // ��ʹ������
			hist,            // ���ֱ��ͼ
			1,                // 1Dֱ��ͼ
			histSize,        // ͳ�ƵĻҶȵĸ���
			ranges);        // �Ҷ�ֵ�ķ�Χ
		return hist;
	}

	Mat getHistogramImage(const Mat &image)
	{
		MatND hist = getHistogram(image);

		// ���ֵ����Сֵ
		double maxVal = 0.0f;
		double minVal = 0.0f;

		minMaxLoc(hist, &minVal, &maxVal);

		//��ʾֱ��ͼ��ͼ��
		Mat histImg(histSize[0], histSize[0], CV_8U, Scalar(255));

		// ������ߵ�Ϊnbins��90%
		int hpt = static_cast<int>(0.9 * histSize[0]);
		//ÿ����Ŀ����һ����ֱ��
		for (int h = 0; h < histSize[0]; h++)
		{
			float binVal = hist.at<float>(h);
			int intensity = static_cast<int>(binVal * hpt / maxVal);
			// ����֮�����һ��ֱ��
			line(histImg, Point(h, histSize[0]), Point(h, histSize[0] - intensity), Scalar::all(0));
		}
		return histImg;
	}

	void mat2GrayHist(Mat &image, int grayHist[64]);

};


class CalcHistogram
{
private:
	int histSize[3];         //ֱ��ͼ�������
	float hranges[2];        //hͨ�����ص���С�����ֵ
	float sranges[2];
	float vranges[2];
	const float *ranges[3];  //��ͨ���ķ�Χ
	int channels[3];         //����ͨ��
	int dims;

public:
	CalcHistogram(int hbins = 4, int sbins = 4, int vbins = 4)
	{
		histSize[0] = hbins;
		histSize[1] = sbins;
		histSize[2] = vbins;
		hranges[0] = 0; hranges[1] = 256;
		sranges[0] = 0; sranges[1] = 256;
		vranges[0] = 0; vranges[1] = 256;
		ranges[0] = hranges;
		ranges[1] = sranges;
		ranges[2] = vranges;
		channels[0] = 0;
		channels[1] = 1;
		channels[2] = 2;
		dims = 3;
	}

	Mat getHistogram(const Mat &image);
	void getHistogramImage(const Mat &image);
	void mat2RGBHist(Mat &image, int hsvHist[64]);

};

float calcCosSimi(int h0[64], int h1[64]);

#endif // CalcHistogram_H
