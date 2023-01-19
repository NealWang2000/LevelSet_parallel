#include "LevelSet.h"
#include <ctime>
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
	cv::Mat src;
	//if (true)
	//{
	//    src = cv::imread("../LevelSet/circle.bmp");
	//    LevelSet levelset(src);
	//    levelset.initializePhi({ 130.0f, 130.0f }, 50.0f);
	//    levelset.evolving();
	//}
	//else
	//{
	clock_t begin, end; // 计时变量
	int k; //收敛次数

	//src = cv::imread("C:/Users/Neal Wang/Desktop/mouth.jpg");
	//LevelSet levelset(src);
	//begin = clock();
	//levelset.initializePhi({ 400.0f, 240.0f }, 200.0f);
	//k = levelset.evolving();
	

			//src = cv::imread("C:/Users/Neal Wang/Desktop/sample/1000.jpg");
			//LevelSet levelset(src);
			//begin = clock();
			//levelset.initializePhi({ 500.0f, 500.0f }, 707.0f);
			//k = levelset.evolving();

	src = cv::imread("C:/Users/Neal Wang/Desktop/plane.jpg");
	begin = clock();
	LevelSet levelset(src);
	levelset.initializePhi({ 160.0f, 285.0f }, 300.0f);
	k = levelset.evolving();

	//src = cv::imread("C:/Users/Neal Wang/Desktop/plane2.jpg");
	//LevelSet levelset(src);
	//begin = clock();
	////levelset.initializePhi({ 400.0f, 400.0f }, 500.0f);
	//levelset.initializePhi({ 290.0f, 400.0f }, 450.0f);
	//k = levelset.evolving();

	//k = levelset.evolving();
	end = clock();
	//levelset.showEvolving();
	cout << "所用时间：" << double(end - begin) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
	cout << "收敛次数为：" << k << endl;
	cv::waitKey(0);
	return EXIT_SUCCESS;
}