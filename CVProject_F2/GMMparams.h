#include"common.h"

class GMM {
public:
	//高斯模型个数
	static const int ncomponent = 5;
	//混合高斯模型的参数初始化
	GMM(Mat& _model);
	//获取像素属于该混合高斯模型的概率
	double operator()(const Vec3d color)const;
	//获取像素属于其中某个高斯模型的概率
	double operator()(int ci, const Vec3d color)const;
	//获取像素属于哪个高斯模型
	int whichComponent(const Vec3d color)const;
	//初始化模型计算的中间变量
	void initLearning();
	//添加像素点
	void addSample(int ci, const Vec3d color);
	//参数计算
	void doLearning();
private:
	//高斯参数模型
	Mat model;
	//每个高斯模型的占比
	double* coefs;
	//每个高斯模型的像素平局值
	double* mean;
	//每个高斯模型的像素协方差矩阵
	double* cov;
	//每个高斯模型的协方差逆矩阵
	double invCov[ncomponent][3][3];
	//每个高斯模型的协方差矩阵的行列式
	double covDet[ncomponent];
	//混合高斯模型建立中间变量
	double sum[ncomponent][3];
	double prod[ncomponent][3][3];
	int nsample[ncomponent];
	int totsample;
private:
	void calcInvCovAndDet(int ci);
};