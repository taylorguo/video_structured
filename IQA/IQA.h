#ifndef IQA_H
#define IQA_H

#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include<algorithm>

#include "CalcHistogram.h"

using namespace cv;
using namespace std;

typedef struct GradInfo_s
{
	float _highMean, _highMean_top10p;
	float _lowMean, _lowMean_top33p;
	float _max, _max_low;
} GradInfo;

typedef struct GradInfo_c_s
{
	float _highMean_10p;
	float _mean;
	float _canny_rate;
} GradInfo_c;

typedef struct Block_8x8_Info_s
{
	Rect _r;
	int _grad_sum;
	float _ssim;
} BlockInfo;

typedef struct NBDiffInfo_s
{
	float _abssum;
	float _absmean;
	float _pix_mean;
	float _pix_sad;
	float _pix_sad_mean;
	float _grad_abssum;
	float _grad_absmean;
} NBDiffInfo;

typedef struct GradInfo2_s
{
	float _gi_h;
	float _gi_v;
} GradInfo2;

typedef struct HSVInfo_s
{
	int _hlo;
	int _hhi;
	int _slo;
	int _shi;
	int _vlo;
	int _vhi;
} HSVInfo;

class CIqa
{
public:
	float calcMos(Mat &img, int bHaveBI);
	float calcIqa(Mat &img)
	{
		Mat img_g;
		cvtColor(img, img_g, CV_BGR2GRAY);

		Mat dif_h;
		getDiff_h(img_g, dif_h);
		float zc_m_h = calcZC_meanvalue_h(dif_h);

		Mat dif_v;
		getDiff_v(img_g, dif_v);
		float zc_m_v = calcZC_meanvalue_v(dif_v);

		cout << zc_m_h << " " << zc_m_v << endl;
		return fmax(zc_m_v, zc_m_h);
	}

	void printIqa(Mat &img);
	void printNRSS(Mat &img);
	void printGradIndex(Mat &img);
	void calcGradInfo2(Mat &g, GradInfo2 *pInfo);
	void printBlockingIndex(Mat &img);
	void printHSVBlurIndex(Mat &img);

	int calcNotZeroCount(Mat &img);
	float calcGradMean(Mat &grad, Mat &otsu, int bHigh);
	void calcGradInfo(GradInfo *pInfo, Mat &grad, Mat &otsu);
	void calcGradInfo_canny(GradInfo_c *pInfo, Mat &grad, Mat &canny_edge);

	float calcReBlurScore(Mat &g);
	float calcGrayGI(Mat &g);

	float calc_fix_index(Mat &g);

private:
	int _noise_thres = 30;
private:
	void mat_merge_bymax(Mat &m0, Mat &m1, Mat &dst);
	void mat_merge_by_sqr(Mat &m0, Mat &m1, Mat &dst);
	float mat_mean_ssd(Mat &m);
	float mat_mean_sad(Mat &m);

	int blocking_count_8x8(Mat &g, Mat &dbg, int x, int y);

	int64 calc_mat_edge_sharp_sum(Mat &g, Mat &edge);
	float calc_mat_edge_sharp_sum2(Mat &g, Mat &blur_g, Mat &edge);
	uchar calc_nb_absdiff(Mat &g, int i, int j);
	void get_grad(Mat &g, Mat &grad);
	void calc_nb_diff_h(Mat &g, Mat &nb_diff);
	void calc_nb_diff_v(Mat &g, Mat &nb_diff);
	void stat_nb_diff(vector<NBDiffInfo> &vInfo, Mat &g, Mat &nb_diff_1p, Mat &nb_diff_2p, int bHer);
	void stat_nb_diff_h(vector<NBDiffInfo> &vInfo, Mat &g, Mat &nb_diff_1p, Mat &nb_diff_2p);
	void stat_nb_diff_v(vector<NBDiffInfo> &vInfo, Mat &g, Mat &nb_diff_1p, Mat &nb_diff_2p);
	void calc_hsvinfo(Mat &img, HSVInfo *info);

	double ssim(Mat &i1, Mat & i2);

	uchar u8max(uchar a, uchar b){
		if (a > b)
			return a;
		else
			return b;
	}

	float fclip(float x, float max, float min) {
		if (x > max)
			return max;
		else if (x<min)
			return min;
		else
			return x;
	}

	float fmax(float a, float b){
		if (a > b)
			return a;
		else
			return b;
	}

	int sabs(short x) {
		if (x > 0)
			return x;
		else
			return -x;
	}

	void getDiff_h(Mat &src, Mat &diff_img)
	{
		int width = src.cols;
		int height = src.rows;

		Rect r_h0, r_h1;
		r_h0 = { 0, 0, width, height - 1 };
		r_h1 = { 0, 1, width, height - 1 };
		Mat src0, src1;
		src0 = src(r_h0);
		src1 = src(r_h1);
		diff_img = Mat(height - 1, width, CV_16SC1);
		for (int i = 0; i < diff_img.rows; i++)
		{
			for (int j = 0; j < diff_img.cols; j++)
			{
				diff_img.at<short>(i, j) = src0.at<uchar>(i, j) - src1.at<uchar>(i, j);
			}
		}
	}

	void getDiff_v(Mat &src, Mat &diff_img)
	{
		int width = src.cols;
		int height = src.rows;

		Rect r_v0, r_v1;
		r_v0 = { 0, 0, width - 1, height };
		r_v1 = { 1, 0, width - 1, height };
		Mat src0, src1;
		src0 = src(r_v0);
		src1 = src(r_v1);
		diff_img = Mat(height, width - 1, CV_16SC1);
		for (int i = 0; i < diff_img.rows; i++)
		{
			for (int j = 0; j < diff_img.cols; j++)
			{
				diff_img.at<short>(i, j) = src0.at<uchar>(i, j) - src1.at<uchar>(i, j);
			}
		}
	}

	float calcZC_meanvalue_h(Mat &diff_img)
	{
		int count = 0;
		float sum = 0;
		for (int i = 0; i < diff_img.rows - 1; i++)
		{
			for (int j = 0; j < diff_img.cols; j++)
			{
				short x = diff_img.at<short>(i, j);
				short y = diff_img.at<short>(i + 1, j);
				float z = x * y;
				if (z < 0)
				{
					if (sabs(x) > _noise_thres && sabs(y) > _noise_thres)
						continue;
				}
				int xy_max = int(fmax(float(sabs(x)), float(sabs(y))));
				if (xy_max< 5)
					continue;

				count++;
				sum += xy_max;
			}
		}
		cout << count << endl;
		return float(sum + 1) / (count + 1);
	}

	float calcZC_meanvalue_v(Mat &diff_img)
	{
		int count = 0;
		float sum = 0;
		for (int i = 0; i < diff_img.rows; i++)
		{
			for (int j = 0; j < diff_img.cols - 1; j++)
			{
				short x = diff_img.at<short>(i, j);
				short y = diff_img.at<short>(i, j + 1);
				float z = x * y;
				if (z < 0)
				{
					if (sabs(x) > _noise_thres && sabs(y) > _noise_thres)
						continue;
				}
				int xy_max = int(fmax(float(sabs(x)), float(sabs(y))));
				if ( xy_max< 10)
					continue;

				count++;
				sum += xy_max;
			}
		}
		cout << count << endl;
		return float(sum + 1) / (count + 1);
	}

	void enhance_equalize(Mat &img, Mat &enh_img)
	{
		Mat imageRGB[3];

		split(img, imageRGB);
		for (int i = 0; i < 3; i++)
		{
			//if (i == 0)
			//	continue;
			equalizeHist(imageRGB[i], imageRGB[i]);
		}
		merge(imageRGB, 3, enh_img);
	}
};

#endif // IQA_H
