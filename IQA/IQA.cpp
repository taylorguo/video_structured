#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include<algorithm>

#include "IQA.h"

using namespace cv;
using namespace std;

static CalcHistogram g_hHist;

#define Labs(x) ((x)>0 ? (x) : -(x))

void CIqa::mat_merge_bymax(Mat &m0, Mat &m1, Mat &dst)
{
	dst = Mat(m0.size(), m0.type());

	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			dst.at<uchar>(i, j) = u8max(m0.at<uchar>(i, j), m1.at<uchar>(i, j));
		}
	}
}

void CIqa::mat_merge_by_sqr(Mat &m0, Mat &m1, Mat &dst)
{
	dst = Mat(m0.size(), m0.type());

	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			int x = m0.at<uchar>(i, j);
			int y = m1.at<uchar>(i, j);
			dst.at<uchar>(i, j) = pow(x*x + y*y, 0.5);
		}
	}
}

float CIqa::mat_mean_ssd(Mat &m)
{
	int i, j;
	int nSum = 0;
	for (i = 0; i < m.rows; i++)
		for (j = 0; j < m.cols; j++)
			nSum += m.at<uchar>(i, j);
	float fMean, fSumSSD = 0;
	fMean = float(nSum) / (m.rows*m.cols);
	for (i = 0; i < m.rows; i++)
		for (j = 0; j < m.cols; j++)
			fSumSSD += (fMean - m.at<uchar>(i, j)) * (fMean - m.at<uchar>(i, j));
	return fSumSSD;
}

float CIqa::mat_mean_sad(Mat &g)
{
	int i, j;
	float fMean, fSumSad = 0;
	fMean = cv::mean(g)[0];
	for (i = 0; i < g.rows; i++)
		for (j = 0; j < g.cols; j++)
			fSumSad += fabs(fMean - g.at<uchar>(i, j));
	return fSumSad;
}

int CIqa::blocking_count_8x8(Mat &g, Mat &dbg, int x, int y)
{
	int count = 0;

	// 边上的一律不是
	if (x == 0 || y == 0 || x >= g.cols - 8 || y >= g.rows - 8)
		return count;

	// 计算内部 sad
	float inner_sad;
	Mat img_mean, dst;
	img_mean = Mat(8, 8, g.type(), mean(g(Rect(x, y, 8, 8))));
	absdiff(g(Rect(x, y, 8, 8)), img_mean, dst);
	inner_sad = cv::sum(dst)[0];

	if (inner_sad > 64 * 2)
		return count;

	// edge sad
	Mat g_tmp;
	vector<float> vEdgeSad, vEdgeSad_inner, vSsd_nb_mean;
	float edge_sad;
	absdiff(g(Rect(x, y - 1, 8, 1)), g(Rect(x, y, 8, 1)), dst);
	edge_sad = cv::sum(dst)[0];
	vEdgeSad.push_back(edge_sad);
	g_tmp = g(Rect(x, y - 1, 8, 1));
	vSsd_nb_mean.push_back(mat_mean_ssd(g_tmp));
	absdiff(g(Rect(x, y + 7, 8, 1)), g(Rect(x, y+8, 8, 1)), dst);
	edge_sad = cv::sum(dst)[0];
	vEdgeSad.push_back(edge_sad);
	g_tmp = g(Rect(x, y + 8, 8, 1));
	vSsd_nb_mean.push_back(mat_mean_ssd(g_tmp));

	absdiff(g(Rect(x - 1, y, 1, 8)), g(Rect(x, y, 1, 8)), dst);
	edge_sad = cv::sum(dst)[0];
	vEdgeSad.push_back(edge_sad);
	g_tmp = g(Rect(x - 1, y, 1, 8));
	vSsd_nb_mean.push_back(mat_mean_ssd(g_tmp));
	
	absdiff(g(Rect(x + 7, y, 1, 8)), g(Rect(x + 8, y, 1, 8)), dst);
	edge_sad = cv::sum(dst)[0];
	vEdgeSad.push_back(edge_sad);
	g_tmp = g(Rect(x + 8, y, 1, 8));
	vSsd_nb_mean.push_back(mat_mean_ssd(g_tmp));

	// 内部边界的 sad
	absdiff(g(Rect(x, y + 1, 8, 1)), g(Rect(x, y, 8, 1)), dst);
	edge_sad = cv::sum(dst)[0];
	vEdgeSad_inner.push_back(edge_sad);
	absdiff(g(Rect(x, y + 7, 8, 1)), g(Rect(x, y + 6, 8, 1)), dst);
	edge_sad = cv::sum(dst)[0];
	vEdgeSad_inner.push_back(edge_sad);
	absdiff(g(Rect(x + 1, y, 1, 8)), g(Rect(x, y, 1, 8)), dst);
	edge_sad = cv::sum(dst)[0];
	vEdgeSad_inner.push_back(edge_sad);
	absdiff(g(Rect(x + 7, y, 1, 8)), g(Rect(x + 6, y, 1, 8)), dst);
	edge_sad = cv::sum(dst)[0];
	vEdgeSad_inner.push_back(edge_sad);

	for (int i = 0; i < 4; i++)
	{
		if (vEdgeSad[i] - vEdgeSad_inner[i]> 28 && vSsd_nb_mean[i]<50*50)
			count++;
	}
	if (count < 2)
	{
		count = 0;
		//dbg(Rect(x, y, 8, 8)) = 0;
	}
	else
		count = 1;

	return count;
}

int CIqa::calcNotZeroCount(Mat &img)
{
	int count = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) != 0)
				count++;
		}
	}
	return count;
}

float CIqa::calcGradMean(Mat &grad, Mat &otsu, int bHigh)
{
	int count = 0;
	int64 grad_sum = 0;
	if (bHigh == 0)
	{
		for (int i = 0; i < grad.rows; i++)
		{
			for (int j = 0; j < grad.cols; j++)
			{
				if (otsu.at<uchar>(i, j) == 0)
				{
					count++;
					grad_sum += pow(grad.at<uchar>(i, j), 1);
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < grad.rows; i++)
		{
			for (int j = 0; j < grad.cols; j++)
			{
				if (otsu.at<uchar>(i, j) != 0)
				{
					count++;
					grad_sum += pow(grad.at<uchar>(i, j), 1);
				}
			}
		}
	}
	return double(grad_sum + 1) / float(count + 1);
}

bool comp_byvalue(const uchar &a, const uchar &b)
{
	return a > b;
}

void CIqa::calcGradInfo(GradInfo *pInfo, Mat &grad, Mat &otsu)
{
	int i, j;
	vector<uchar> vHigh, vLow;
	for ( i = 0; i < grad.rows; i++)
	{
		for ( j = 0; j < grad.cols; j++)
		{
			if (otsu.at<uchar>(i, j) != 0)
				vHigh.push_back(grad.at<uchar>(i, j));
			else
				vLow.push_back(grad.at<uchar>(i, j));
		}
	}
	sort(vHigh.begin(), vHigh.end(), comp_byvalue);
	sort(vLow.begin(), vLow.end(), comp_byvalue);
	
	int64 high_sum = 0, low_sum = 0;
	int64 high_sum_10p = 0, low_sum_33p = 0;
	for (i = 0; i < vHigh.size(); i++)
		high_sum += vHigh[i];
	for (i = 0; i < vLow.size(); i++)
		low_sum += vLow[i];
	for (i = 0; i < vHigh.size() / 10; i++)
		high_sum_10p += vHigh[i];
	for (i = 0; i < vLow.size() / 3; i++)
		low_sum_33p += vLow[i];
	pInfo->_highMean = double(high_sum) / (vHigh.size() + 1);
	pInfo->_highMean_top10p = double(high_sum_10p) / (vHigh.size() / 10 + 1);
	pInfo->_lowMean = double(low_sum) / (vLow.size() + 1);
	pInfo->_lowMean_top33p = double(low_sum_33p) / (vLow.size() / 3 + 1);
	if (vHigh.size()>0)
		pInfo->_max = vHigh[0];
	else
		pInfo->_max = 0;
	pInfo->_max_low = vLow[0];
}

void CIqa::calcGradInfo_canny(GradInfo_c *pInfo, Mat &grad, Mat &canny_edge)
{
	int i, j;
	vector<uchar> vPix_c;
	for (i = 0; i < grad.rows; i++)
	{
		for (j = 0; j < grad.cols; j++)
		{
			if (canny_edge.at<uchar>(i, j) != 0)
				vPix_c.push_back(grad.at<uchar>(i, j));
		}
	}
	if (vPix_c.size() == 0)
	{
		pInfo->_mean = 0;
		pInfo->_highMean_10p = 0;
		return;
	}
	sort(vPix_c.begin(), vPix_c.end(), comp_byvalue);
	
	int64 sum_c = 0;
	int64 high_sum_10p = 0;
	for (i = 0; i < vPix_c.size()/10; i++)
		high_sum_10p += vPix_c[i];
	for (i = 0; i < vPix_c.size(); i++)
		sum_c += vPix_c[i];
	pInfo->_mean = double(sum_c) / vPix_c.size();
	if (vPix_c.size() / 10 == 0)
		pInfo->_highMean_10p = 0;
	else
		pInfo->_highMean_10p = double(high_sum_10p) / (vPix_c.size() / 10);

	pInfo->_canny_rate = float(vPix_c.size()) / (canny_edge.rows*canny_edge.cols);
}

uchar CIqa::calc_nb_absdiff(Mat &g, int i, int j)
{
	uchar ad = 0;

	uchar t0, t1, t2;
	uchar c0, c1, c2;
	uchar b0, b1, b2;
	t0 = g.at<uchar>(i - 1, j - 1);
	t1 = g.at<uchar>(i - 1, j);
	t2 = g.at<uchar>(i - 1, j + 1);
	c0 = g.at<uchar>(i, j - 1);
	c1 = g.at<uchar>(i, j);
	c2 = g.at<uchar>(i, j + 1);
	b0 = g.at<uchar>(i + 1, j - 1);
	b1 = g.at<uchar>(i + 1, j);
	b2 = g.at<uchar>(i + 1, j + 1);
#if 0
	uchar t;
	t = (t0 + t2 + b0 + b2 + 3 * t1 + 3 * c0 + 3 * c2 + 3 * b1 + 8) >> 4;
	ad = Labs(c1 - t);
#else
	vector<uchar> vPix;
	vPix.push_back(g.at<uchar>(i - 1, j - 1));
	vPix.push_back(g.at<uchar>(i - 1, j));
	vPix.push_back(g.at<uchar>(i - 1, j + 1));
	vPix.push_back(g.at<uchar>(i, j - 1));
	vPix.push_back(g.at<uchar>(i, j));
	vPix.push_back(g.at<uchar>(i, j + 1));
	vPix.push_back(g.at<uchar>(i + 1, j - 1));
	vPix.push_back(g.at<uchar>(i + 1, j));
	vPix.push_back(g.at<uchar>(i + 1, j + 1));
	sort(vPix.begin(), vPix.end(), comp_byvalue);
	uchar t = g.at<uchar>(i, j);
	ad = u8max(Labs(t-vPix[0]), Labs(t-vPix[8]));
#endif
	return ad;
}

float CIqa::calc_mat_edge_sharp_sum2(Mat &g, Mat &blur_g, Mat &edge)
{
	int i, j;
	int64 es_sum = 0, es_sum_blur = 0;
	int count = 0;

	for (i = 1; i < edge.rows - 1; i++)
	{
		for (j = 1; j < edge.cols - 1; j++)
		{
			if (edge.at<uchar>(i, j) != 0)
			{
				uchar ad_g = calc_nb_absdiff(g, i, j);
				uchar ad_g_blur = calc_nb_absdiff(blur_g, i, j);
				if (ad_g > ad_g_blur)
				{
					es_sum += ad_g;
					es_sum_blur += ad_g_blur;
					count++;
				}
			}
		}
	}
	//cout << "count111 : " << count << endl;
	return double(es_sum - es_sum_blur + 1) / (es_sum + 1);
}

int64 CIqa::calc_mat_edge_sharp_sum(Mat &g, Mat &edge)
{
	int i, j;
	int64 es_sum = 0;

	for (i = 1; i < edge.rows-1; i++)
	{
		for (j = 1; j < edge.cols-1; j++)
		{
			if (edge.at<uchar>(i, j) != 0)
			{
				uchar t0, t1, t2;
				uchar c0, c1, c2;
				uchar b0, b1, b2;
				t0 = g.at<uchar>(i - 1, j - 1);
				t1 = g.at<uchar>(i - 1, j);
				t2 = g.at<uchar>(i - 1, j + 1);
				c0 = g.at<uchar>(i, j - 1);
				c1 = g.at<uchar>(i, j);
				c2 = g.at<uchar>(i, j + 1);
				b0 = g.at<uchar>(i + 1, j - 1);
				b1 = g.at<uchar>(i + 1, j);
				b2 = g.at<uchar>(i + 1, j + 1);

				uchar t;
				t = (t0 + t2 + b0 + b2 + 3 * t1 + 3 * c0 + 3 * c2 + 3 * b1 + 8) >> 4;
				es_sum += Labs(c1 - t);
			}
		}
	}

	return es_sum;
}

float CIqa::calcReBlurScore(Mat &g)
{
	Mat canny_edge, reblur;
	int64 src_ess, blur_ess;
	Canny(g, canny_edge, 5, 15);
	
	GaussianBlur(g, reblur, Size(3, 3), 0);

	//imshow("reblur", sobel_edge);
	//imshow("src", g);
	//waitKey();

	return calc_mat_edge_sharp_sum2(g, reblur, canny_edge);
}

void CIqa::get_grad(Mat &g, Mat &grad)
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	//Scharr(g, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	Sobel(g, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	//Scharr(g, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	Sobel(g, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	mat_merge_bymax(abs_grad_x, abs_grad_y, grad);
}

void CIqa::calc_nb_diff_h(Mat &g, Mat &nb_diff)
{
	int d = CV_16S;
	Mat g_16s;
	g.convertTo(g_16s, d);

	nb_diff = Mat(Size(g.cols-1, g.rows), CV_16SC1);

	nb_diff = g_16s(Rect(1, 0, g_16s.cols - 1, g_16s.rows)) - g_16s(Rect(0, 0, g_16s.cols - 1, g_16s.rows));
}

void CIqa::calc_nb_diff_v(Mat &g, Mat &nb_diff)
{
	int d = CV_16S;
	Mat g_16s;
	g.convertTo(g_16s, d);

	nb_diff = Mat(Size(g.cols, g.rows-1), CV_16SC1);

	nb_diff = g_16s(Rect(0, 1, g_16s.cols, g_16s.rows - 1)) - g_16s(Rect(0, 0, g_16s.cols, g_16s.rows - 1));
}

void CIqa::stat_nb_diff_h(vector<NBDiffInfo> &vInfo, Mat &g, Mat &nb_diff_1p, Mat &nb_diff_2p)
{
	int i;
	for (i = 0; i < nb_diff_2p.rows; i++)
	{
		NBDiffInfo info;
		info._abssum = sum(cv::abs(nb_diff_2p(Rect(0, i, nb_diff_2p.cols, 1))))[0];
		info._absmean = info._abssum / nb_diff_2p.cols;

		info._grad_abssum = sum(cv::abs(nb_diff_1p(Rect(0, i, nb_diff_1p.cols, 1))))[0];
		info._grad_absmean = info._grad_abssum / nb_diff_1p.cols;
		
		int pix_sum = 0;
		pix_sum = sum(g(Rect(0, i, g.cols, 1)))[0];
		info._pix_mean = double(pix_sum) / g.cols;

		Mat img_mean = Mat(Size(g.cols, 1), g.type());;
		img_mean = info._pix_mean + 0.5;
		Mat dst;
		absdiff(g(Rect(0, i, g.cols, 1)), img_mean, dst);
		info._pix_sad = sum(dst)[0];
		info._pix_sad_mean = info._pix_sad / g.cols;

		vInfo.push_back(info);
	}
}

void CIqa::stat_nb_diff_v(vector<NBDiffInfo> &vInfo, Mat &g, Mat &nb_diff_1p, Mat &nb_diff_2p)
{
	int i;
	for (i = 0; i < nb_diff_2p.cols; i++)
	{
		NBDiffInfo info;
		info._abssum = sum(cv::abs(nb_diff_2p(Rect(i, 0, 1, nb_diff_2p.rows))))[0];
		info._absmean = info._abssum / nb_diff_2p.rows;

		info._grad_abssum = sum(cv::abs(nb_diff_1p(Rect(i, 0, 1, nb_diff_1p.rows))))[0];
		info._grad_absmean = info._grad_abssum / nb_diff_1p.rows;

		int pix_sum = 0;
		pix_sum = sum(g(Rect(i, 0, 1, g.rows)))[0];
		info._pix_mean = double(pix_sum) / g.rows;

		Mat img_mean = Mat(Size(1, g.rows), g.type());;
		img_mean = info._pix_mean + 0.5;
		Mat dst;
		absdiff(g(Rect(i, 0, 1, g.rows)), img_mean, dst);
		info._pix_sad = sum(dst)[0];
		info._pix_sad_mean = info._pix_sad / g.rows;

		vInfo.push_back(info);
	}
}

void CIqa::stat_nb_diff(vector<NBDiffInfo> &vInfo, Mat &g, Mat &nb_diff_1p, Mat &nb_diff_2p, int bHer)
{
	if (bHer == 0)
		stat_nb_diff_v(vInfo, g, nb_diff_1p, nb_diff_2p);
	else
		stat_nb_diff_h(vInfo, g, nb_diff_1p, nb_diff_2p);
}

void CIqa::calc_hsvinfo(Mat &img, HSVInfo *pInfo)
{
	int i, j;
	pInfo->_hhi = pInfo->_shi = pInfo->_vhi = 0;
	pInfo->_hlo = pInfo->_slo = pInfo->_vlo = 255;
	for (i = 0; i < img.rows; i++)
	{
		uchar *p = img.ptr<uchar>(i);
		for (j = 0; j < img.cols; j++)
		{
			int h, s, v;
			h = p[3 * j + 0];
			s = p[3 * j + 1];
			v = p[3 * j + 2];
			if (h > pInfo->_hhi)
				pInfo->_hhi = h;
			if (h < pInfo->_hlo)
				pInfo->_hlo = h;
			if (s > pInfo->_shi)
				pInfo->_shi = s;
			if (s < pInfo->_slo)
				pInfo->_slo = s;
			if (v > pInfo->_vhi)
				pInfo->_vhi = v;
			if (v < pInfo->_vlo)
				pInfo->_vlo = v;
		}
	}
}

double CIqa::ssim(Mat &i1, Mat & i2)
{
	const Size gs_size = { 5, 5 };
	float x_sigma = 1.5;
	const double C1 = 6.5025, C2 = 58.5225;
	int d = CV_32F;
	Mat I1, I2;
	i1.convertTo(I1, d);
	i2.convertTo(I2, d);
	Mat I1_2 = I1.mul(I1);
	Mat I2_2 = I2.mul(I2);
	Mat I1_I2 = I1.mul(I2);
	Mat mu1, mu2;
	GaussianBlur(I1, mu1, gs_size, x_sigma);
	GaussianBlur(I2, mu2, gs_size, x_sigma);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigam2_2, sigam12;
	GaussianBlur(I1_2, sigma1_2, gs_size, x_sigma);
	sigma1_2 -= mu1_2;

	GaussianBlur(I2_2, sigam2_2, gs_size, x_sigma);
	sigam2_2 -= mu2_2;

	GaussianBlur(I1_I2, sigam12, gs_size, x_sigma);
	sigam12 -= mu1_mu2;
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigam12 + C2;
	t3 = t1.mul(t2);

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigam2_2 + C2;
	t1 = t1.mul(t2);

	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar mssim = mean(ssim_map);

	if (i1.channels() == 1)
		return mssim.val[0];
	else
	{
		double ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3;
		return ssim;
	}
}

void CIqa::printIqa(Mat &img)
{
	Mat g;
	Rect r = { 0, img.rows / 6, img.cols, img.rows*4/6 };
	cvtColor(img, g, CV_BGR2GRAY);

	Mat grad, grad_blur;
	Mat canny_edge, reblur;

	get_grad(g, grad);
	blur(g, reblur, Size(3, 3));
	get_grad(reblur, grad_blur);

	Canny(g, canny_edge, 10, 30);

	Mat otsu, otsu_blur;
	threshold(grad, otsu, 255, 255, THRESH_OTSU);
	threshold(grad_blur, otsu_blur, 255, 255, THRESH_OTSU);

#if 0
	imshow("src", img);
	imshow("grad", grad);
	imshow("ots", otsu);
	imshow("canny", canny_edge);
	imshow("reblur", reblur);
	waitKey();
#endif
	int pixnum = grad.rows*grad.cols;
	int grad_notzero = calcNotZeroCount(grad);
	int highcount = calcNotZeroCount(otsu);
	int lowcount = pixnum - highcount;
	GradInfo info, info_blur;
	calcGradInfo(&info, grad, otsu);
	calcGradInfo(&info_blur, grad_blur, otsu_blur);

	GradInfo_c info_c;
	calcGradInfo_canny(&info_c, grad, canny_edge);
	//float highmean = calcGradMean(grad, otsu, 1);
	//float lowmean = calcGradMean(grad, otsu, 0);

	float fReblurScore;
	fReblurScore = calcReBlurScore(g);

	cout << float(grad_notzero) / pixnum << ",";
	cout << float(highcount) / pixnum << "," << float(calcNotZeroCount(otsu_blur)) / pixnum << "," << info._highMean << "," << info._highMean_top10p << ",";
	cout << float(lowcount) / pixnum << "," << info._lowMean << "," << info._lowMean_top33p << ",";
	cout << info._max << "," << info._max_low << ",";
	//cout << info_c._highMean_10p << "," << info_c._mean << "," << info_c._canny_rate << ",";
	//cout << fReblurScore << ",";
	cout << info_blur._highMean << "," << info_blur._highMean_top10p << ",";
	cout << info_blur._lowMean << "," << info_blur._lowMean_top33p << "," << info_blur._max << "," << info_blur._max_low;
}

bool comp_by_sum(const BlockInfo &a, const BlockInfo &b)
{
	return a._ssim < b._ssim;
}


void CIqa::printNRSS(Mat &img)
{
#ifdef WIN32
	int i, j;
	Mat g, g_blur;
	Mat grad, grad_blur, otsu;

	cvtColor(img, g, CV_BGR2GRAY);
	get_grad(g, grad);
	blur(g, g_blur, Size(5, 5));
	threshold(grad, otsu, 255, 255, THRESH_OTSU);

	// 计算 8*8 块的 grad sum
	vector<BlockInfo> vBlock;
	for (i = 0; i < img.rows / 8; i++)
	{
		for (j = 0; j < img.cols / 8; j++)
		{
			BlockInfo bi;
			bi._r = { j * 8, i * 8, 8, 8 };

			if (sum(otsu(bi._r))[0] < 255 * 8)
				continue;

			bi._grad_sum = sum(grad(bi._r))[0];
			bi._ssim = ssim(g(bi._r), g_blur(bi._r));
			vBlock.push_back(bi);
		}
	}
	sort(vBlock.begin(), vBlock.end(), comp_by_sum);

	int N = 64;
	if (N > vBlock.size())
		N = vBlock.size();

	float ssim_sum = 0;
	for (i = 0; i < N; i++)
	{
		cout << vBlock[i]._ssim << " ";
		ssim_sum += vBlock[i]._ssim;
	}

	float nrss = (N == 0 ? 0 : (1 - ssim_sum / N));
	cout << nrss;
	cout << " " << ssim(g, g_blur);
#endif
}

void CIqa::calcGradInfo2(Mat &g, GradInfo2 *pInfo)
{
	Mat diff_1p_h, diff_1p_v, diff_2p_h, diff_2p_v;

	calc_nb_diff_h(g, diff_1p_h);
	calc_nb_diff_v(g, diff_1p_v);
	calc_nb_diff_h(diff_1p_h, diff_2p_h);
	calc_nb_diff_v(diff_1p_v, diff_2p_v);

	vector<NBDiffInfo> vNBInfo_h, vNBInfo_v;
	stat_nb_diff(vNBInfo_h, g, diff_1p_h, diff_2p_h, 1);
	stat_nb_diff(vNBInfo_v, g, diff_1p_v, diff_2p_v, 0);

	double pix_sum_h = 0, grad_sum_h = 0, grad_2p_sum_h = 0;
	for (int i = 0; i < vNBInfo_h.size(); i++)
	{
		pix_sum_h += vNBInfo_h[i]._pix_sad_mean;
		grad_sum_h += vNBInfo_h[i]._grad_absmean;
		grad_2p_sum_h += vNBInfo_h[i]._absmean;
	}
	double pix_sum_v = 0, grad_sum_v = 0, grad_2p_sum_v = 0;
	for (int i = 0; i < vNBInfo_v.size(); i++)
	{
		pix_sum_v += vNBInfo_v[i]._pix_sad_mean;
		grad_sum_v += vNBInfo_v[i]._grad_absmean;
		grad_2p_sum_v += vNBInfo_v[i]._absmean;
	}

	pInfo->_gi_h = (grad_2p_sum_h) / (vNBInfo_h.size());
	pInfo->_gi_v = (grad_2p_sum_v) / (vNBInfo_v.size());
	//pInfo->_gi_h = (grad_sum_h*grad_2p_sum_h) / (vNBInfo_h.size()*vNBInfo_h.size());
	//pInfo->_gi_v = (grad_sum_v*grad_2p_sum_v) / (vNBInfo_v.size()*vNBInfo_v.size());

	//cout << pix_sum_h / vNBInfo_h.size() << "," << grad_sum_h / vNBInfo_h.size() << "," << grad_2p_sum_h / vNBInfo_h.size() << ",";
	//cout << pix_sum_v / vNBInfo_v.size() << "," << grad_sum_v / vNBInfo_v.size() << "," << grad_2p_sum_v / vNBInfo_v.size();
}

float CIqa::calcGrayGI(Mat &g)
{
	GradInfo2 info;
	calcGradInfo2(g, &info);

	Mat b;
	blur(g, b, Size(3, 3));
	GradInfo2 info_b;
	calcGradInfo2(b, &info_b);

	float score_gi;
	float score_h, score_v;
	if (info._gi_h == 0)
		score_h = 1;
	else
		score_h = info_b._gi_h / info._gi_h;
	if (info._gi_v == 0)
		score_v = 1;
	else
		score_v = info_b._gi_v / info._gi_v;
	score_gi = 1 - (score_h < score_v ? score_h : score_v);

	return score_gi;
}

float CIqa::calc_fix_index(Mat &g)
{
	float fi = 1.0;
	Mat grad;
	//Canny(g, edge, 5, 30);
	get_grad(g, grad);
	threshold(grad, grad, 10, 255, THRESH_OTSU);
	//imshow("grad", grad);
	//waitKey();

	float mean_g =  mean(g)[0];
	float edge_rate = float(calcNotZeroCount(grad)) / (g.cols*g.rows);
	//cout << mat_mean_sad(g) / (g.cols*g.rows) << endl;
	//cout << mean_g << " " << edge_rate << endl;
	if (mean_g < 100.0)
		fi = (mean_g+50) / 150;
	if (mean_g<80 && edge_rate > 0.24)
		fi = fi*0.67;

	return fi;
}

void CIqa::printGradIndex(Mat &img)
{
	Mat g, diff_1p_h, diff_1p_v, diff_2p_h, diff_2p_v;
	cvtColor(img, g, CV_BGR2GRAY);

	GradInfo2 info;
	calcGradInfo2(g, &info);

	Mat b;
	blur(g, b, Size(3, 3));
	GradInfo2 info_b;
	calcGradInfo2(b, &info_b);
	
	float score_h, score_v, score;
	score_h = info_b._gi_h / info._gi_h;
	score_v = info_b._gi_v / info._gi_v;
	score = score_h < score_v ? score_h : score_v;
	cout << 1 - pow(score, 1);
}

void CIqa::printBlockingIndex(Mat &img)
{
	int i, j;

	int count = 0;
	Mat g, g_dbg;
	cvtColor(img, g, CV_BGR2GRAY);
	g_dbg = g.clone();
	for (i = 0; i < img.rows; i += 8)
	{
		for (j = 0; j < img.cols; j += 8)
		{
			int bBlocking;
			bBlocking = blocking_count_8x8(g, g_dbg, j, i);
			count += bBlocking;
		}
	}
	//cout << count << " ";
	float score_bi;
	score_bi = float(count) / ((img.rows - 8)*(img.cols - 8) / 64 + 1);
	cout << pow(score_bi, 0.5);
	imwrite("dbg.jpg", g_dbg);
}

void CIqa::printHSVBlurIndex(Mat &img)
{
	Mat enh_img;
	enhance_equalize(img, enh_img);

	Mat img128, enh_128;
	resize(img, img128, Size(128, 128));
	resize(enh_img, enh_128, Size(128, 128));
	//cvtColor(img128, img128, CV_BGR2HSV);
	//cvtColor(enh_128, enh_128, CV_BGR2HSV);

	HSVInfo info, info_enh;
	calc_hsvinfo(img128, &info);
	calc_hsvinfo(enh_128, &info_enh);

	cout << info._hhi << "," << info._hlo << "," << info._shi << "," << info._slo << ",";
	cout << info._vhi << "," << info._vlo;

	Mat imageHSV[3];
	split(img128, imageHSV);

	Histogram1D hist1d;
	Mat hsvhist;
	hsvhist = hist1d.getHistogram(imageHSV[1]);
	cout << " " << hsvhist << endl;
#if 0
	int hsvHist0[64], hsvHist1[64];
	g_hHist.mat2RGBHist(img128, hsvHist0);
	g_hHist.mat2RGBHist(enh_128, hsvHist1);
	for (int i = 0; i < 64; i++)
	{
		cout << hsvHist0[i] << " " << hsvHist1[i] << endl;
	}
	cout << calcCosSimi(hsvHist0, hsvHist1);
#endif
}

float CIqa::calcMos(Mat &img, int bHaveBI)
{
	// 先计算 blur gi
	Mat g_dbg;
	Mat g, diff_1p_h, diff_1p_v, diff_2p_h, diff_2p_v;
	cvtColor(img, g, CV_BGR2GRAY);
	if (g.rows >= 720 && g.cols>200)
		g(Rect(g.cols - 180, 0, 180, 70)) = mean(g(Rect(g.cols - 180, 0, 180, 70)));

	float fi;
	fi = calc_fix_index(g);

	float score_gi, score_gi_d8, score_bi;
	score_gi = calcGrayGI(g);
	//cout << "gi : " << score_gi << endl;
	
	Mat g_d8;
	resize(g, g_d8, Size(g.cols / 8, g.rows / 8));
	score_gi_d8 = calcGrayGI(g_d8);
	//cout << score_gi_d8 << " " << score_gi << endl;
	if (score_gi_d8 < score_gi) // 有噪声
	{
		blur(g, g, Size(5, 5));
		score_gi = calcGrayGI(g)*0.5;
	}
	score_gi *= fi;

	//cout << "gi : " << score_gi << endl;

	int i, j, count = 0;
	//if (bHaveBI != 0)
	{
		//g_dbg = g.clone();
		for (i = 0; i < img.rows; i += 8)
		{
			for (j = 0; j < img.cols; j += 8)
			{
				int block_count;
				block_count = blocking_count_8x8(g, g_dbg, j, i);
				count += block_count;
			}
		}
		score_bi = 1 - 9 * float(count - 2) / ((img.rows - 8)*(img.cols - 8) / 64 + 1);
		score_bi = 0.2 + score_bi*0.8;
		//cout << count << "," << score_bi << ",";
		//imshow("dbg", g_dbg);
		//waitKey();
	}
	//else
	//	score_bi = 0;
	//cout << score_bi << " " << score_gi  << endl;
	float mos = score_gi * score_bi;

	return mos;
}
