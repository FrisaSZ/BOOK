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

//void HistoMatch()

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

	cvWaitKey(0);
}