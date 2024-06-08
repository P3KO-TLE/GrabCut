#include"common.h"

class GMM {
public:
	//��˹ģ�͸���
	static const int ncomponent = 5;
	//��ϸ�˹ģ�͵Ĳ�����ʼ��
	GMM(Mat& _model);
	//��ȡ�������ڸû�ϸ�˹ģ�͵ĸ���
	double operator()(const Vec3d color)const;
	//��ȡ������������ĳ����˹ģ�͵ĸ���
	double operator()(int ci, const Vec3d color)const;
	//��ȡ���������ĸ���˹ģ��
	int whichComponent(const Vec3d color)const;
	//��ʼ��ģ�ͼ�����м����
	void initLearning();
	//������ص�
	void addSample(int ci, const Vec3d color);
	//��������
	void doLearning();
private:
	//��˹����ģ��
	Mat model;
	//ÿ����˹ģ�͵�ռ��
	double* coefs;
	//ÿ����˹ģ�͵�����ƽ��ֵ
	double* mean;
	//ÿ����˹ģ�͵�����Э�������
	double* cov;
	//ÿ����˹ģ�͵�Э���������
	double invCov[ncomponent][3][3];
	//ÿ����˹ģ�͵�Э������������ʽ
	double covDet[ncomponent];
	//��ϸ�˹ģ�ͽ����м����
	double sum[ncomponent][3];
	double prod[ncomponent][3][3];
	int nsample[ncomponent];
	int totsample;
private:
	void calcInvCovAndDet(int ci);
};