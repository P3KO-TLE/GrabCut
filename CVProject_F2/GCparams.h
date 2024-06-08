#include"GMMparams.h"

class GcParam {
private:
	//ԭͼָ��
	const Mat* src;
	//ԭͼ��С
	int rows, cols;
	//�ɰ棺����ǰ��������ǩ
	Mat mask;
	//ǰ�����������ĸ�˹ģ�ͱ�ǩ
	Mat compId;
	//�������ڱ�Ȩ�������
	double beta, gamma, lambda;
	//�������ڱ�Ȩ
	Mat leftC, upleftC, upC, uprightC;
	//ǰ�������Ļ�ϸ�˹ģ��
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