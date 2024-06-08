#include"GMMparams.h"

class GcParam {
private:
	//原图指针
	const Mat* src;
	//原图大小
	int rows, cols;
	//蒙版：储存前景背景标签
	Mat mask;
	//前景（背景）的高斯模型标签
	Mat compId;
	//像素相邻边权计算参数
	double beta, gamma, lambda;
	//像素相邻边权
	Mat leftC, upleftC, upC, uprightC;
	//前景背景的混合高斯模型
	Mat fModel, bModel;
	GMM fgdGMM = GMM(fModel), bgdGMM = GMM(bModel);
public:
	void init(const Mat* src);
	void calcBeta();
	void calcNCost();
	void initMask(Rect rect);
	void initGMM();
	void assignGMMComponents();
	void learnGMM();
	double getNCost(int r, int c, int tag);
	double getTCost(const Vec3b& color, int tag);
	int getFBTag(int r, int c);
	void setFgd(int r, int c);
	void setBgd(int r, int c);
};