#include"GCparams.h"

void GcParam::init(const Mat* src) {
	cout << "	GcParam:init开始\n";
	this->src = src;
	rows = src->rows, cols = src->cols;
	this->beta = 0, this->gamma = 50, this->lambda = 100;
	compId.create(rows, cols, CV_32S);
	compId.setTo(Scalar(0));
	leftC.create(rows, cols, CV_64FC1);
	upleftC.create(rows, cols, CV_64FC1);
	upC.create(rows, cols, CV_64FC1);
	uprightC.create(rows, cols, CV_64FC1);
	mask.create(rows, cols, CV_8UC1);
	cout << "	GcParam:init结束\n";
}

void GcParam::calcBeta() {
	cout << "	GcParam:calcBeta开始\n";
	beta = 0;
	for (int y = 0; y < rows; y++)for (int x = 0; x < cols; x++) {
		Vec3d color = src->at<Vec3b>(y, x);
		if (x > 0) {
			Vec3d diff = color - (Vec3d)src->at<Vec3b>(y, x - 1);
			beta += diff.dot(diff);
		}
		if (y > 0 && x > 0) {
			Vec3d diff = color - (Vec3d)src->at<Vec3b>(y - 1, x - 1);
			beta += diff.dot(diff);
		}
		if (y > 0) {
			Vec3d diff = color - (Vec3d)src->at<Vec3b>(y - 1, x);
			beta += diff.dot(diff);
		}
		if (y > 0 && x < cols - 1) {
			Vec3d diff = color - (Vec3d)src->at<Vec3b>(y - 1, x + 1);
			beta += diff.dot(diff);
		}
	}

	if (beta <= numeric_limits<double>::epsilon())
		beta = 0;
	else
		beta = 1.0 / (2 * beta / (4 * cols * rows - 3 * cols - 3 * rows + 2));

	cout << "	GcParam:calcBeta结束\n";
}

void GcParam::calcNCost() {
	cout << "	GcParam:calcNCost开始\n";
	const double gamma1 = gamma / sqrt(2.0);

	for (int y = 0; y < rows; y++)for (int x = 0; x < cols; x++) {
		Vec3d color = src->at<Vec3b>(y, x);
		if (x > 0) {
			Vec3d diff = color - (Vec3d)src->at<Vec3b>(y, x - 1);
			leftC.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
		}
		else
			leftC.at<double>(y, x) = 0;
		if (x > 0 && y > 0) {
			Vec3d diff = color - (Vec3d)src->at<Vec3b>(y - 1, x - 1);
			upleftC.at<double>(y, x) = gamma1 * exp(-beta * diff.dot(diff));
		}
		else
			upleftC.at<double>(y, x) = 0;
		if (y > 0) {
			Vec3d diff = color - (Vec3d)src->at<Vec3b>(y - 1, x);
			upC.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
		}
		else
			upC.at<double>(y, x) = 0;
		if (y > 0 && x < cols - 1) {
			Vec3d diff = color - (Vec3d)src->at<Vec3b>(y - 1, x + 1);
			uprightC.at<double>(y, x) = gamma1 * exp(-beta * diff.dot(diff));
		}
		else
			uprightC.at<double>(y, x) = 0;
	}
	cout << "	GcParam:calcNCost结束\n";
}

void GcParam::initMask(Rect rect) {
	cout << "	GcParam:initMask开始\n";
	mask.setTo(GC_BGD);
	(mask(rect)).setTo(Scalar(GC_PR_FGD));
	cout << "	GcParam:initMask结束\n";
}

void GcParam::initGMM() {
	cout << "	GcParam:initGMM开始\n";
	const int kMeansItCount = 20;
	const int kMeansType = KMEANS_PP_CENTERS;

	Mat bgdLabels, fgdLabels;
	vector<Vec3f>bgdSamples, fgdSamples;
	for (int y = 0; y < rows; y++)for (int x = 0; x < cols; x++) {
		if (mask.at<uchar>(y, x) == GC_BGD || mask.at<uchar>(y, x) == GC_PR_BGD)
			bgdSamples.push_back((Vec3f)src->at<Vec3b>(y, x));
		else
			fgdSamples.push_back((Vec3f)src->at<Vec3b>(y, x));
	}
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	kmeans(_bgdSamples, GMM::ncomponent, bgdLabels, TermCriteria(TermCriteria::COUNT, kMeansItCount, 0.0), 0, kMeansType);
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, GMM::ncomponent, fgdLabels, TermCriteria(TermCriteria::COUNT, kMeansItCount, 0.0), 0, kMeansType);

	bgdGMM.initLearning();
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.doLearning();

	fgdGMM.initLearning();
	for (int i = 0; i < (int)fgdSamples.size(); i++) 
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.doLearning();
	cout << "	GcParam:initGMM结束\n";
}

void GcParam::assignGMMComponents() {
	cout << "	GcParam:assignGMMComponents开始\n";
	for (int y = 0; y < rows; y++)for (int x = 0; x < cols; x++) {
		Vec3d color = src->at<Vec3b>(y, x);
		compId.at<int>(y, x) = (mask.at<uchar>(y, x) == GC_BGD || mask.at<uchar>(y, x) == GC_PR_BGD) ? bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
	}
	cout << "	GcParam:assignGMMComponents结束\n";
}

void GcParam::learnGMM() {
	cout << "	GcParamlearnGMM开始\n";
	bgdGMM.initLearning();
	fgdGMM.initLearning();
	for (int y = 0; y < rows; y++)for (int x = 0; x < cols; x++) {
		int ci = compId.at<int>(y, x);
		if (mask.at<uchar>(y, x) == GC_BGD || mask.at<uchar>(y, x) == GC_PR_BGD)
			bgdGMM.addSample(ci, src->at<Vec3b>(y, x));
		else
			fgdGMM.addSample(ci, src->at<Vec3b>(y, x));
	}
	bgdGMM.doLearning();
	fgdGMM.doLearning();
	cout << "	GcParamlearnGMM结束\n";
}

double GcParam::getNCost(int r, int c, int tag) {
	switch (tag)
	{
	case 0:
		return leftC.at<double>(r, c);
	case 1:
		return upleftC.at<double>(r, c);
	case 2:
		return upC.at<double>(r, c);
	case 3:
		return uprightC.at<double>(r, c);
	default:
		return 0;
	}
}

double GcParam::getTCost(const Vec3b& color, int tag) {
	switch (tag)
	{
	case 0:
		return -log(bgdGMM(color));
	case 1:
		return -log(fgdGMM(color));
	default:
		return 0;
	}
}

int GcParam::getFBTag(int r, int c) {
	return (int)mask.at<uchar>(r, c);
}

void GcParam::setFgd(int r, int c) {
	mask.at<uchar>(r, c) = GC_PR_FGD;
}

void GcParam::setBgd(int r, int c) {
	mask.at<uchar>(r, c) = GC_PR_BGD;
}