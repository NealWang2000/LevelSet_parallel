#include "LevelSet.h"
#include <stdio.h>

using namespace cv;
using namespace std;

LevelSet::LevelSet(const cv::Mat& src)
{
	if (src.channels() == 3)
	{
		cv::cvtColor(src, src_, COLOR_BGR2GRAY);
		src.copyTo(image_);
	}
	else
	{
		src.copyTo(src_);
		cv::cvtColor(src, image_, COLOR_GRAY2BGR);
	}
	src_.convertTo(src_, CV_32FC1);
	phi_ = Mat::zeros(src_.size(), CV_32FC1);
	dirac_ = Mat::zeros(src_.size(), CV_32FC1);
	heaviside_ = Mat::zeros(src_.size(), CV_32FC1);
}

// 初始化水平集
void LevelSet::initializePhi(cv::Point2f center, float radius)
{
	const float c = 2.0f;
	float value = 0.0;
	for (int i = 0; i < src_.rows; i++)
	{
		for (int j = 0; j < src_.cols; j++)
		{
			value = -sqrt(pow((j - center.x), 2) + pow((i - center.y), 2)) + radius;
			if (abs(value) < 1e-3)
			{
				//在零水平集曲线上
				phi_.at<float>(i, j) = 0;
			}
			else
			{
				// 在零水平集内：为正
				// 在零水平集外：为负
				phi_.at<float>(i, j) = value;
			}

		}
	}
}

// 计算梯度
void LevelSet::gradient(const cv::Mat& src, cv::Mat& gradx, cv::Mat& grady)
{
	if (src.rows < 2 || src.cols < 2)
		return;
	src.copyTo(gradx);
	src.copyTo(grady);
	// 水平方向梯度
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (j == 0)
				gradx.at<float>(i, j) = src.at<float>(i, j + 1) - src.at<float>(i, j);
			else if (j == src.cols - 1)
				gradx.at<float>(i, j) = src.at<float>(i, j) - src.at<float>(i, j - 1);
			else
				gradx.at<float>(i, j) = (src.at<float>(i, j + 1) - src.at<float>(i, j - 1)) / 2.0;
			if (gradx.at<float>(i, j) == 0)
			{
				gradx.at<float>(i, j) = 1e-8;
			}
		}
	}
	// 垂直方向梯度
	for (int j = 0; j < src.cols; j++)
	{
		for (int i = 0; i < src.rows; i++)
		{
			if (i == 0)
				grady.at<float>(i, j) = src.at<float>(i + 1, j) - src.at<float>(i, j);
			else if (i == src.rows - 1)
				grady.at<float>(i, j) = src.at<float>(i, j) - src.at<float>(i - 1, j);
			else
				grady.at<float>(i, j) = (src.at<float>(i + 1, j) - src.at<float>(i - 1, j)) / 2.0;
			if (grady.at<float>(i, j) == 0)
			{
				grady.at<float>(i, j) = 1e-8;
			}
		}
	}

}

// 计算dirac level set: δ(φ)
void LevelSet::dirac()
{
	const float k1 = epsilon_ / CV_PI;
	const float k2 = epsilon_ * epsilon_;
	for (int i = 0; i < src_.rows; i++)
	{
		for (int j = 0; j < src_.cols; j++)
		{
			dirac_.at<float>(i, j) = k1 / (k2 + pow(phi_.at<float>(i, j), 2));
		}
	}
}

// 计算heaviside：Н(φ)
// 本质为阶跃函数，添加epsilon_项用于平滑处理
// 其可以运用阶跃性值对前景和背景进行区分
void LevelSet::heaviside()
{
	const float k3 = 2 / CV_PI;
	for (int i = 0; i < src_.rows; i++)
	{
		for (int j = 0; j < src_.cols; j++)
		{
			heaviside_.at<float>(i, j) = 0.5 * (1.0 + k3 * atan(phi_.at<float>(i, j) / epsilon_));
		}
	}
	Mat temp = heaviside_.clone();
}

//  计算曲率
void LevelSet::calculateCurvature()
{
	Mat dx, dy;
	gradient(src_, dx, dy);
	Mat norm = Mat::zeros(src_.size(), CV_32FC1);
	for (int i = 0; i < src_.rows; i++)
	{
		for (int j = 0; j < src_.cols; j++)
		{
			norm.at<float>(i, j) = pow(dx.at<float>(i, j) * dx.at<float>(i, j) + dy.at<float>(i, j) * dy.at<float>(i, j), 0.5);
		}
	}
	Mat dxx, dxy, dyx, dyy;
	gradient(dx / norm, dxx, dxy);
	gradient(dy / norm, dyx, dyy);
	curv_ = dxx + dyy;
}

// 计算像素点前景与背景均值
void LevelSet::calculatC()
{
	c1_ = 0.0f;
	c2_ = 0.0f;
	float sum1 = 0.0f;
	float h1 = 0.0f;
	float sum2 = 0.0f;
	float h2 = 0.0f;
	float value = 0.0f;
	float h = 0.0f;
	for (int i = 0; i < src_.rows; i++)
	{
		for (int j = 0; j < src_.cols; j++)
		{
			value = src_.at<float>(i, j);
			h = heaviside_.at<float>(i, j);

			h1 += h;
			sum1 += h * value;
			h2 += (1 - h);
			sum2 += (1 - h) * value;
		}
	}
	//sum1 = h * value;
	c1_ = forntpro_ * sum1 / (h1 + 1e-10);
	c2_ = sum2 / (h2 + 1e-10);
	//printf("c1:%e,c2:%e\n", c1_, c2_);
}

// 绘制演化过程
void LevelSet::showEvolving()
{
	Mat image = image_.clone();
	Mat mask = phi_ >= 0;
	Mat phic = phi_;
	cv::dilate(mask, mask, Mat(), Point(-1, -1), 3);
	cv::erode(mask, mask, Mat(), Point(-1, -1), 3);
	vector<vector<Point> > contours;
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	drawContours(image, contours, -1, CV_RGB(0, 255, 0), 2);
	namedWindow("Evolving");
	moveWindow("Evolving", 600, 100);
	imshow("Evolving", image);
	waitKey(1);
}

// 绘制水平集演化图
void LevelSet::showLevelsetEvolving()
{
	Mat phic = phi_.clone();
	Mat mask = phi_ >= 0;
	Mat phinom;
	Mat phinomu;
	Mat phiev; // 水平集函数演化展示图像
	normalize(phic, phinom, 0, 255, NORM_MINMAX, -2, mask);
	phinom.convertTo(phinomu, CV_8UC1);
	applyColorMap(phinomu, phiev, COLORMAP_JET);
	namedWindow("EvolvingLevelset");
	moveWindow("EvolvingLevelset", 122, 100);
	//namedWindow("EvolvingLevelset");
	//moveWindow("EvolvingLevelset", 282, 142);
	imshow("EvolvingLevelset", phiev);
	waitKey(1);
}

// 具体演化函数
// 运用迭代法解偏微分方程，求解φ对应时间间隔的变化量
int LevelSet::evolving()
{
	//showEvolving();
	//showLevelsetEvolving();
	// 迭代次数
	int k;
	bool flag; //是否收敛判定标识
	//iteration
	for (k = 0; k < iterationnum_; k++)
	{
		flag = false;
		heaviside();
		dirac();
		calculatC();
		calculateCurvature();
		//update phi
		// 模拟演化过程
		for (int i = 0; i < src_.rows; i++)
		{
			for (int j = 0; j < src_.cols; j++)
			{
				float curv = curv_.at<float>(i, j);
				float dirac = dirac_.at<float>(i, j);
				float u0 = src_.at<float>(i, j);

				float lengthTerm = mu_ * dirac * curv;
				float areamterm = nu_ * dirac;
				float fittingterm = dirac * (-lambda1_ * pow(u0 - c1_, 2) + lambda2_ * pow(u0 - c2_, 2));
				float term = lengthTerm + areamterm + fittingterm;
				float phinew = phi_.at<float>(i, j) + timestep_ * term;
				float phiold = phi_.at<float>(i, j);
				phi_.at<float>(i, j) = phinew;

				if (!flag)
				{
					if (phinew * phiold < 0)
					{
						flag = true;
					}
				}

			}
		}
		//showEvolving();
		//showLevelsetEvolving();
		// 对是否收敛进行判断
		if (!flag)
		{
			break;
		}

	}
	showEvolving();
	showLevelsetEvolving();
	return k;
}
