#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

void HistoEq(Mat &input, Mat &output) // 作为提醒，这里使用double型，像素值需要在[0, 1]上
{
	int hist[256];
	double cdHist[256];
	memset(hist, 0, 256 * sizeof(int));
	memset(cdHist, 0, 256 * sizeof(double));

	int nRows, nCols;
	nRows = input.rows;
	nCols = input.cols;

	for (int i = 0; i < nRows; ++i)
	{
		uchar *p = input.ptr<uchar>(i);
		for (int j = 0; j < nCols; ++j)
		{
			hist[p[j]]++;
		}
	}

	int pixelNum = nRows * nCols;
	for (int i = 0; i < 256; ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			cdHist[i] += hist[j];
		}
		cdHist[i] /= pixelNum;
	}

	for (int i = 0; i < nRows; ++i)
	{
		uchar *p = input.ptr<uchar>(i);
		double *q = output.ptr<double>(i);
		for (int j = 0; j < nCols; ++j)
		{
			q[j] = cdHist[p[j]];
		}
	}

}

void HistoMatch(Mat &input, Mat &output) // 这次使用uchar型
{

	int hist[256], cdHist[256], cdHist_out[256], inv_cdHist[256];
	memset(hist, 0, 256 * sizeof(int));
	memset(cdHist, 0, 256 * sizeof(int));
	memset(cdHist_out, 0, 256 * sizeof(int));
	memset(inv_cdHist, -1, 256 * sizeof(int));

	int nRows, nCols;
	nRows = input.rows;
	nCols = input.cols;

	for (int i = 0; i < nRows; ++i)
	{
		uchar *p = input.ptr<uchar>(i);
		for (int j = 0; j < nCols; ++j)
		{
			hist[p[j]]++;
		}
	}

	int pixelNum = nRows * nCols;
	for (int i = 0; i < 256; ++i)
	{
		double freq = 0;
		for (int j = 0; j <= i; ++j)
		{
			freq += hist[j];
		}
		freq /= pixelNum;
		cdHist[i] = (int)(0.5 + freq * 255); // 四舍五入转成整型
	}

	// 下面求cdHist[]的不严格逆变换
	for (int i = 0; i < 256; ++i)
	{
		// 因为cdHist[]不是一一映射，会出现cdHist[i] = cdHist[i + 1]的情况， 
		// 所以inv_cdHist[]的同一个位置可能被写多次，而有些位置却没有被写过，形成间断点，值为初始化时的-1
		inv_cdHist[cdHist[i]] = i; 									
	}
	//待填补间断点，方法是直接找到最近的非间断点，使用同一个值
	int i, j;
	i = j = 0;
	while (i < 255)
	{
		if (inv_cdHist[i + 1] != -1)
		{
			++i;
			continue;
		}
		j = 1;
		while (inv_cdHist[i + j] == -1 && (i + j) <= 255)
		{
			inv_cdHist[i + j] = inv_cdHist[i];
			++j;
		}
	}

	//计算output的均衡化变换
	memset(hist, 0, 256 * sizeof(int));
	for (int i = 0; i < nRows; ++i)
	{
		uchar *p = output.ptr<uchar>(i);
		for (int j = 0; j < nCols; ++j)
		{
			hist[p[j]]++;
		}
	}

	for (int i = 0; i < 256; ++i)
	{
		double freq = 0;
		for (int j = 0; j <= i; ++j)
		{
			freq += hist[j];
		}
		freq /= pixelNum;
		cdHist_out[i] = (int)(0.5 + freq * 255); // 四舍五入转成整型
	}

	for (int i = 0; i < nRows; ++i)
	{
		uchar *p = output.ptr<uchar>(i);
		for (int j = 0; j < nCols; ++j)
		{
			p[j] = inv_cdHist[cdHist_out[p[j]]];
		}
	}
}

void main(int argc, char **argv)
{
	Mat img = imread(argv[1]);
	vector<Mat> BGR, BGR_EQ;
	split(img, BGR);
	
	int nWidth, nHeight;
	nWidth = img.cols;
	nHeight = img.rows;
	Mat EQ_B(nHeight, nWidth, CV_64FC1),
		EQ_G(nHeight, nWidth, CV_64FC1),
		EQ_R(nHeight, nWidth, CV_64FC1), 
		img_EQ;
	
	imshow("Origin", img);

	HistoEq(BGR[0], EQ_B);
	HistoEq(BGR[1], EQ_G);
	HistoEq(BGR[2], EQ_R);

	BGR_EQ.push_back(EQ_B);
	BGR_EQ.push_back(EQ_G);
	BGR_EQ.push_back(EQ_R);
	merge(BGR_EQ, img_EQ);
	imshow("img_EQ", img_EQ);
	imshow("before_match", BGR[1]);
	Mat tmp = BGR[1].clone();
	HistoMatch(BGR[0], tmp);
	imshow("after_match", tmp);

	cvWaitKey(0);
}