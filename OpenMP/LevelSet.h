
#pragma once
#include "opencv2/opencv.hpp"

class LevelSet
{
public:
	LevelSet(const cv::Mat& src);
	//initialize level set
	void initializePhi(cv::Point2f center, float radius);
	int evolving();

private:
	int iterationnum_ = 1000;
	float lambda1_ = 1.0f;	 //weight for c1 fitting term
	float lambda2_ = 1.0f;   //weight for c2 fitting term
	float mu_ = 0.1 * 255 * 255;	//μ: weight for length term长度项
	float nu_ = 0.0;  //ν: weight for area term, default value 0 面积项
	float timestep_ = 5; //time step: δt
	//float timestep_ = 1; //time step: δt
	//ε: parameter for computing smooth Heaviside and dirac function
	float epsilon_ = 1.0;
	float c1_;	//average(u0) inside level set
	float c2_;	//average(u0) outside level set
	float forntpro_ = 0.1; //前景均值调节系数
	cv::Mat phi_;		//level set: φ
	cv::Mat dirac_;		//dirac level set: δ(φ)
	cv::Mat heaviside_;	//heaviside：Н(φ)
	cv::Mat curv_;
	cv::Mat src_;
	cv::Mat image_; //for showing显示图像

	void gradient(const cv::Mat& src, cv::Mat& gradx, cv::Mat& grady);
	//Dirac function
	void dirac();
	//Heaviside function
	void heaviside();
	void calculateCurvature();
	//calculate c1 and c2
	void calculatC();
	//show evolving
	void showEvolving();
	void showLevelsetEvolving();
};

