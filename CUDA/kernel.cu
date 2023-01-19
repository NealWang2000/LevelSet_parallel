#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <cmath>
#include <algorithm>   
using namespace cv;
using namespace std;
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error %s (%d) at %s:%d\n", cudaGetErrorString(x),x, __FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

// 块大小定义
#define BLOCKWIDTH 16
#define BLOCKHEIGHT 16

// 全局变量存储区域
Mat phi_;		//level set: φ
Mat src_;
Mat image_; //for showing显示图像
const int iterationnum_ = 30000;



// GPU常量存储区域
//__constant__ int d_iterationnum = 500;
__constant__ float d_lambda1 = 1.0f;
__constant__ float d_lambda2 = 1.0f;
__constant__ float d_mu = 0.1 * 255 * 255;	//长度项
__constant__ float d_nu = 0.0;  //面积项
//__constant__ float d_timestep = 1; //δt 嘴唇步长
__constant__ float d_timestep = 5; //δt 飞机步长
//__constant__ float d_timestep = 5; //δt 性能步长
__constant__ float d_epsilon = 1.0;
__constant__ float d_k1 = 1 / CV_PI; // d_epsilon=1的情况
__constant__ float d_k2 = 1.0; // d_epsilon=1的情况
__constant__ float d_k3 = 2 / CV_PI; // d_epsilon=1的情况
__constant__ float d_forntpro = 0.1; //前景均值调节系数

// 飞机2符号距离函数初始化参数
//__constant__ float d_phi0centerx = 290.0; //初始符号距离函数中心x
//__constant__ float d_phi0centery = 400.0; //初始符号距离函数中心y
//__constant__ float d_phi0r = 450.0; //初始符号距离函数半径

// 飞机1符号距离函数初始化参数
__constant__ float d_phi0centerx = 160.0; //初始符号距离函数中心x
__constant__ float d_phi0centery = 285.0; //初始符号距离函数中心y
__constant__ float d_phi0r = 300.0; //初始符号距离函数半径

// 嘴唇符号距离函数初始化参数
//__constant__ float d_phi0centerx = 400.0; //初始符号距离函数中心x
//__constant__ float d_phi0centery = 240.0; //初始符号距离函数中心y
//__constant__ float d_phi0r = 200.0; //初始符号距离函数半径

// 性能符号距离函数初始化参数2
//__constant__ float d_phi0centerx = 250.0; //初始符号距离函数中心x
//__constant__ float d_phi0centery = 250.0; //初始符号距离函数中心y
//__constant__ float d_phi0r = 353.0; //初始符号距离函数半径


// GPU全局变量存储区域
__device__ float d_c1;
__device__ float d_c2;
__device__ float h1result;
__device__ float h2result;
__device__ float sum1result;
__device__ float sum2result;



// 计算x方向与y方向梯度
__device__	void gradient(float* d_src, float* gradx, float* grady, int idx, int idy, uint imgWidth, uint imgHeight)
{
	float xtemp = 0, ytemp = 0;
 
	// 水平方向梯度
	if (idx == 0)
	{
		xtemp = d_src[idy * imgWidth + (idx + 1)] - d_src[idy * imgWidth + idx];
	}
	else if (idx == imgWidth - 1)
	{
		xtemp = d_src[idy * imgWidth + idx] - d_src[idy * imgWidth + idx - 1];
	}
	else
	{
		xtemp = (d_src[idy * imgWidth + (idx + 1)] - d_src[idy * imgWidth + idx - 1]) / 2.0;
	}

	if (xtemp == 0)
	{
		gradx[idy * imgWidth + idx] = 1e-8;
	}
	else
	{
		gradx[idy * imgWidth + idx] = xtemp;
	}


	// 垂直方向梯度
	if (idy == 0)
	{
		ytemp = d_src[(idy + 1) * imgWidth + idx] - d_src[idy * imgWidth + idx];
	}
	else if (idy == imgHeight - 1)
	{
		ytemp = d_src[idy * imgWidth + idx] - d_src[(idy - 1) * imgWidth + idx];
	}
	else
	{
		ytemp = (d_src[(idy + 1) * imgWidth + idx] - d_src[(idy - 1) * imgWidth + idx]) / 2.0;
	}

	if (ytemp == 0)
	{
		grady[idy * imgWidth + idx] = 1e-8;
	}
	else
	{
		grady[idy * imgWidth + idx] = ytemp;
	}
}

// 计算x方向梯度
__device__	void gradientx(float* d_src, float* gradx, int idx, int idy, uint imgWidth, uint imgHeight)
{
	float xtemp = 0;

	// 水平方向梯度
	if (idx == 0)
	{
		xtemp = d_src[idy * imgWidth + (idx + 1)] - d_src[idy * imgWidth + idx];
	}
	else if (idx == imgWidth - 1)
	{
		xtemp = d_src[idy * imgWidth + idx] - d_src[idy * imgWidth + idx - 1];
	}
	else
	{
		xtemp = (d_src[idy * imgWidth + (idx + 1)] - d_src[idy * imgWidth + idx - 1]) / 2.0;
	}

	if (xtemp == 0)
	{
		gradx[idy * imgWidth + idx] = 1e-8;
	}
	else
	{
		gradx[idy * imgWidth + idx] = xtemp;
	}
}


// 计算y方向梯度
__device__	void gradienty(float* d_src, float* grady, int idx, int idy, uint imgWidth, uint imgHeight)
{
	float ytemp = 0;

	// 垂直方向梯度
	if (idy == 0)
	{
		ytemp = d_src[(idy + 1) * imgWidth + idx] - d_src[idy * imgWidth + idx];
	}
	else if (idy == imgHeight - 1)
	{
		ytemp = d_src[idy * imgWidth + idx] - d_src[(idy - 1) * imgWidth + idx];
	}
	else
	{
		ytemp = (d_src[(idy + 1) * imgWidth + idx] - d_src[(idy - 1) * imgWidth + idx]) / 2.0;
	}

	if (ytemp == 0)
	{
		grady[idy * imgWidth + idx] = 1e-8;
	}
	else
	{
		grady[idy * imgWidth + idx] = ytemp;
	}
}


// zong二维归并函数
__device__ void gpuReduction2zong(float* d_a, float* d_value, uint imgWidth, uint imgHeight)
{
	__shared__ float temph1[BLOCKHEIGHT][BLOCKWIDTH], temph2[BLOCKHEIGHT][BLOCKWIDTH], tempsum1[BLOCKHEIGHT][BLOCKWIDTH], tempsum2[BLOCKHEIGHT][BLOCKWIDTH];
	// idx 为列号，纵坐标
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// idy 为行号，横坐标
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idx < imgWidth && idy < imgHeight)
	{
		int ycrow = imgHeight % BLOCKHEIGHT;
		int yccol = imgWidth % BLOCKWIDTH;
		// 待归约变量提取
		float h1per = d_a[idy * imgWidth + idx];
		//printf("%f\n", h1per);
		float h2per = 1 - d_a[idy * imgWidth + idx];
		float sum1per = d_a[idy * imgWidth + idx] * d_value[idy * imgWidth + idx];
		float sum2per = (1 - d_a[idy * imgWidth + idx]) * d_value[idy * imgWidth + idx];

		// 当存在块超出图片的情况
		if (ycrow > 0 || yccol > 0)
		{
			int ycblocky = imgHeight / BLOCKHEIGHT;
			int ycblockx = imgWidth / BLOCKWIDTH;

			if ((blockIdx.x > (ycblockx - 1)) || (blockIdx.y > (ycblocky - 1)))
			{
				atomicAdd(&h1result, h1per);
				atomicAdd(&h2result, h2per);
				atomicAdd(&sum1result, sum1per);
				atomicAdd(&sum2result, sum2per);

				//printf("%d,%d\n", blockIdx.x, blockIdx.y);
			}
			else
			{
				temph1[threadIdx.y][threadIdx.x] = h1per;
				temph2[threadIdx.y][threadIdx.x] = h2per;
				tempsum1[threadIdx.y][threadIdx.x] = sum1per;
				tempsum2[threadIdx.y][threadIdx.x] = sum2per;

				__syncthreads();

				// 对行进行规约
				for (int i = (BLOCKWIDTH >> 1); i > 0; i >>= 1) {
					if (threadIdx.x < i) {
						temph1[threadIdx.y][threadIdx.x] += temph1[threadIdx.y][threadIdx.x + i];
						temph2[threadIdx.y][threadIdx.x] += temph2[threadIdx.y][threadIdx.x + i];
						tempsum1[threadIdx.y][threadIdx.x] += tempsum1[threadIdx.y][threadIdx.x + i];
						tempsum2[threadIdx.y][threadIdx.x] += tempsum2[threadIdx.y][threadIdx.x + i];
					}
					__syncthreads();
				}
				//__syncthreads();
				//printf("%f\n", temp[k][0]);
				// 对列进行规约
				if (threadIdx.x == 0)
				{
					for (int i = (BLOCKHEIGHT >> 1); i > 0; i >>= 1) {
						if (threadIdx.y < i) {
							temph1[threadIdx.y][threadIdx.x] += temph1[threadIdx.y + i][threadIdx.x];
							temph2[threadIdx.y][threadIdx.x] += temph2[threadIdx.y + i][threadIdx.x];
							tempsum1[threadIdx.y][threadIdx.x] += tempsum1[threadIdx.y + i][threadIdx.x];
							tempsum2[threadIdx.y][threadIdx.x] += tempsum2[threadIdx.y + i][threadIdx.x];
						}
						__syncthreads();
					}
				}
				__syncthreads();
				// 全局规约结果汇总
				if (threadIdx.x == 0 && threadIdx.y == 0)
				{
					atomicAdd(&h1result, temph1[0][0]);
					atomicAdd(&h2result, temph2[0][0]);
					atomicAdd(&sum1result, tempsum1[0][0]);
					atomicAdd(&sum2result, tempsum2[0][0]);
				}
			}
		}
		// 当不存在块超出图片的情况
		else
		{
			temph1[threadIdx.y][threadIdx.x] = h1per;
			temph2[threadIdx.y][threadIdx.x] = h2per;
			tempsum1[threadIdx.y][threadIdx.x] = sum1per;
			tempsum2[threadIdx.y][threadIdx.x] = sum2per;

			__syncthreads();

			// 对行进行规约
			for (int i = (BLOCKWIDTH >> 1); i > 0; i >>= 1) {
				if (threadIdx.x < i) {
					temph1[threadIdx.y][threadIdx.x] += temph1[threadIdx.y][threadIdx.x + i];
					temph2[threadIdx.y][threadIdx.x] += temph2[threadIdx.y][threadIdx.x + i];
					tempsum1[threadIdx.y][threadIdx.x] += tempsum1[threadIdx.y][threadIdx.x + i];
					tempsum2[threadIdx.y][threadIdx.x] += tempsum2[threadIdx.y][threadIdx.x + i];
				}
				__syncthreads();
			}
			//__syncthreads();
			//printf("%f\n", temp[k][0]);
			// 对列进行规约
			if (threadIdx.x == 0)
			{
				for (int i = (BLOCKHEIGHT >> 1); i > 0; i >>= 1) {
					if (threadIdx.y < i) {
						temph1[threadIdx.y][threadIdx.x] += temph1[threadIdx.y + i][threadIdx.x];
						temph2[threadIdx.y][threadIdx.x] += temph2[threadIdx.y + i][threadIdx.x];
						tempsum1[threadIdx.y][threadIdx.x] += tempsum1[threadIdx.y + i][threadIdx.x];
						tempsum2[threadIdx.y][threadIdx.x] += tempsum2[threadIdx.y + i][threadIdx.x];
					}
					__syncthreads();
				}
				//printf("%f\n", temp[0][0]);
			}
			__syncthreads();
			// 全局规约结果汇总
			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				atomicAdd(&h1result, temph1[0][0]);
				atomicAdd(&h2result, temph2[0][0]);
				atomicAdd(&sum1result, tempsum1[0][0]);
				atomicAdd(&sum2result, tempsum2[0][0]);
			}
		}
	}
}


__global__ void evolvingArg(float* d_phi, float* d_dirac, float* d_heaviside, float* d_curv, float* d_src, float* d_dx, float* d_dy, float* d_dxx,
	float* d_dyy, uint imgWidth, uint imgHeight) {

	// idx 为列号，纵坐标
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// idy 为行号，横坐标
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < imgWidth && idy < imgHeight)
	{
		d_dirac[idy * imgWidth + idx] = d_k1 / (d_k2 + powf(d_phi[idy * imgWidth + idx], 2));
		d_heaviside[idy * imgWidth + idx] = 0.5 * (1.0 + d_k3 * atanf(d_phi[idy * imgWidth + idx] / d_epsilon));
		// 曲率计算
		gradient(d_src, d_dx, d_dy, idx, idy, imgWidth, imgHeight);
		float norm = powf(d_dx[idy * imgWidth + idx] * d_dx[idy * imgWidth + idx] + d_dy[idy * imgWidth + idx] * d_dy[idy * imgWidth + idx], 0.5);
		d_dx[idy * imgWidth + idx] = d_dx[idy * imgWidth + idx] / norm;
		d_dy[idy * imgWidth + idx] = d_dy[idy * imgWidth + idx] / norm;
	}
	else
	{
		return;
	}

}



__global__ void evolvingCrvAndAvg(float* d_phi, float* d_dirac, float* d_heaviside, float* d_curv, float* d_src, float* d_dx, float* d_dy, float* d_dxx,
	float* d_dyy, uint imgWidth, uint imgHeight) {
	// idx 为列号，纵坐标
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// idy 为行号，横坐标
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < imgWidth && idy < imgHeight)
	{
		gradientx(d_dx, d_dxx, idx, idy, imgWidth, imgHeight);
		gradienty(d_dy, d_dyy, idx, idy, imgWidth, imgHeight);
		d_curv[idy * imgWidth + idx] = d_dxx[idy * imgWidth + idx] + d_dyy[idy * imgWidth + idx];

		float value = d_src[idy * imgWidth + idx];
		float h = d_heaviside[idy * imgWidth + idx];
		if (idx == 0 && idy == 0)
		{
			h1result = 0;
			h2result = 0;
			sum1result = 0;
			sum2result = 0;
		}
		__threadfence();
		gpuReduction2zong(d_heaviside, d_src, imgWidth, imgHeight);

	}
	else
	{
		return;
	}

}
__global__ void evolvingProAndCheck(float* d_phi, float* d_dirac, float* d_heaviside, float* d_curv, float* d_src, float* d_dx, float* d_dy, float* d_dxx,
	float* d_dyy, int* d_flag, uint imgWidth, uint imgHeight) {
	// idx 为列号，纵坐标
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// idy 为行号，横坐标
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	//float c1, c2;
	if (idx == 0 && idy == 0)
	{
		d_c1 = d_forntpro * sum1result / (h1result + 1e-10);
		d_c2 = sum2result / (h2result + 1e-10);
		*d_flag = 0;
		//printf("h1:%e,h2:%e,sum1:%e,sum2:%e,c1:%e,c2:%e\n", h1result,h2result,sum1result,sum2result, c1, c2);
	}
	__threadfence();
	if (idx < imgWidth && idy < imgHeight)
	{
		float curv = d_curv[idy * imgWidth + idx];
		float dirac = d_dirac[idy * imgWidth + idx];
		float u0 = d_src[idy * imgWidth + idx];

		float lengthTerm = d_mu * dirac * curv;
		float areamterm = d_nu * dirac;
		float fittingterm = dirac * (-d_lambda1 * powf(u0 - d_c1, 2) + d_lambda2 * powf(u0 - d_c2, 2));
		float term = lengthTerm + areamterm + fittingterm;
		float newphi = d_phi[idy * imgWidth + idx] + d_timestep * term;
		float oldphi = d_phi[idy * imgWidth + idx];
		d_phi[idy * imgWidth + idx] = newphi;
		if (*d_flag == 0)
		{
			if (oldphi * newphi < 0)
			{
				*d_flag = 1;
			}
			__threadfence();
		}
	}
	else
	{
		return;
	}

}

__global__ void cudaInitializePhi(float* d_phi, uint imgWidth, uint imgHeight) {
	// idx 为列号，纵坐标
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// idy 为行号，横坐标
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idx < imgWidth && idy < imgHeight)
	{
		float value = -sqrtf(powf((idx - d_phi0centerx), 2) + powf((idy - d_phi0centery), 2)) + d_phi0r;
		if (abs(value) < 1e-3)
		{
			//在零水平集曲线上
			d_phi[idy * imgWidth + idx] = 0;
		}
		else
		{
			// 在零水平集内：为正
			// 在零水平集外：为负
			d_phi[idy * imgWidth + idx] = value;
		}
	}
	else
	{
		return;
	}

}


// 绘制演化过程
void showEvolving()
{
	Mat image = image_.clone();
	Mat mask = phi_ >= 0;
	// 进行开运算
	cv::dilate(mask, mask, Mat(), Point(-1, -1), 3);
	cv::erode(mask, mask, Mat(), Point(-1, -1), 3);
	vector<vector<Point>> contours;
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	drawContours(image, contours, -1, CV_RGB(0, 255, 0), 2);
	//namedWindow("Evolving");
	//moveWindow("Evolving", 600, 100);
	imshow("Evolving", image);
	waitKey(1);
}

// 绘制水平集演化图
void showLevelsetEvolving()
{
	Mat phic = phi_.clone();
	Mat mask = phi_ >= 0;
	Mat phinom;
	Mat phinomu;
	Mat phiev; // 水平集函数演化展示图像
	normalize(phic, phinom, 0, 255, NORM_MINMAX, -2, mask);
	phinom.convertTo(phinomu, CV_8UC1);
	applyColorMap(phinomu, phiev, COLORMAP_JET);
	//namedWindow("EvolvingLevelset");
	//moveWindow("EvolvingLevelset", 122, 100);
	imshow("EvolvingLevelset", phiev);
	waitKey(1);
}

int main()
{
	Mat src;
	//src = imread("C:/Users/Neal Wang/Desktop/mouth.jpg");
	src = imread("C:/Users/Neal Wang/Desktop/plane.jpg");
	//src = imread("C:/Users/Neal Wang/Desktop/plane2.jpg");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	if (src.channels() == 3)
	{
		cv::cvtColor(src, src_, COLOR_BGR2GRAY);
		src.copyTo(image_);
	}
	else if (src.channels() == 1)
	{
		src.copyTo(src_);
		cv::cvtColor(src, image_, COLOR_GRAY2BGR);
	}
	else
	{
		printf("请输入合理通道数图片");
	}
	// 参数初始化
	src_.convertTo(src_, CV_32FC1);
	phi_ = Mat::zeros(src_.size(), CV_32FC1);

	int imgHeight = src_.rows;
	int imgWidth = src_.cols;

	// 全局参数定义
	float* d_phi;
	float* d_dirac;
	float* d_heaviside;
	float* d_curv;
	float* d_src;
	// 曲率计算相关参数
	float* d_dx, * d_dy, * d_dxx, * d_dyy;
	// 是否收敛标识参数
	int* d_flag;

	CUDA_CALL(cudaMalloc((void**)&d_phi, imgHeight * imgWidth * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_dirac, imgHeight * imgWidth * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_heaviside, imgHeight * imgWidth * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_curv, imgHeight * imgWidth * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_src, imgHeight * imgWidth * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_dx, imgHeight * imgWidth * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_dy, imgHeight * imgWidth * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_dxx, imgHeight * imgWidth * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_dyy, imgHeight * imgWidth * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_flag, sizeof(int)));

	// 数据传输
	CUDA_CALL(cudaMemcpy(d_src, src_.data, imgHeight * imgWidth * sizeof(float), cudaMemcpyHostToDevice));

	// block与grid设置
	dim3 block1(BLOCKWIDTH, BLOCKHEIGHT);
	dim3 grid1((imgWidth + block1.x - 1) / block1.x, (imgHeight + block1.y - 1) / block1.y);

	int flag = 1;
	int convnum = iterationnum_;
	// 初始化符号距离函数
	cudaInitializePhi << <grid1, block1 >> > (d_phi, imgWidth, imgHeight);
	// 具体分割演化过程
	for (int i = 0; i < iterationnum_; i++)
	{
		// 迭代次数传输
		evolvingArg << <grid1, block1 >> > (d_phi, d_dirac, d_heaviside, d_curv, d_src, d_dx, d_dy, d_dxx, d_dyy, imgWidth, imgHeight);

		evolvingCrvAndAvg << <grid1, block1 >> > (d_phi, d_dirac, d_heaviside, d_curv, d_src, d_dx, d_dy, d_dxx, d_dyy, imgWidth, imgHeight);

		evolvingProAndCheck << <grid1, block1 >> > (d_phi, d_dirac, d_heaviside, d_curv, d_src, d_dx, d_dy, d_dxx, d_dyy, d_flag, imgWidth, imgHeight);

		CUDA_CALL(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
		if (flag == 0)
		{
			convnum = i;
			break;
		}
		//CUDA_CALL(cudaMemcpy(phi_.data, d_phi, imgHeight * imgWidth * sizeof(float), cudaMemcpyDeviceToHost));
		//showEvolving();
		//showLevelsetEvolving();
	}
	CUDA_CALL(cudaMemcpy(phi_.data, d_phi, imgHeight * imgWidth * sizeof(float), cudaMemcpyDeviceToHost));
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("所用时间：%3.1f ms\n", elapsedTime);
	printf("收敛次数：%d", convnum);

	showEvolving();
	showLevelsetEvolving();
	// 释放预先分配的显存
	cudaFree(d_phi);
	cudaFree(d_dirac);
	cudaFree(d_heaviside);
	cudaFree(d_curv);
	cudaFree(d_src);
	cudaFree(d_dx);
	cudaFree(d_dy);
	cudaFree(d_dxx);
	cudaFree(d_dyy);
	cudaFree(d_flag);



	waitKey(0);
	return 0;
}
