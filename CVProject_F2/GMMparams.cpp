#include"GMMparams.h"

GMM::GMM(Mat& _model) {
	const int size = 3 + 9 + 1;
	
	_model.create(1, size, CV_64FC1);
	_model.setTo(Scalar(0));
	

	model = _model;
	coefs = model.ptr < double>(0);
	mean = coefs + ncomponent;
	cov = mean + ncomponent * 3;

	for (int i = 0; i < ncomponent; i++)
		if (coefs[i] > 0)
			calcInvCovAndDet(i);
}

double GMM::operator()(const Vec3d color)const {
	double res = 0;
	for (int i = 0; i < ncomponent; i++)
		res += coefs[i] * (*this)(i, color);
	return res;
}

double GMM::operator()(int ci, const Vec3d color)const {
	if (coefs[ci] > 0) {
		Vec3d diff = color;
		double* m = mean + 3 * ci;
		diff[0] -= m[0], diff[1] -= m[1], diff[2] -= m[2];
		double mul = diff[0] * (diff[0] * invCov[ci][0][0] + diff[1] * invCov[ci][1][0] + diff[2] * invCov[ci][2][0])
			+ diff[1] * (diff[0] * invCov[ci][0][1] + diff[1] * invCov[ci][1][1] + diff[2] * invCov[ci][2][1])
			+ diff[2] * (diff[0] * invCov[ci][0][2] + diff[1] * invCov[ci][1][2] + diff[2] * invCov[ci][2][2]);
		return 1.0 / sqrt(covDet[ci]) * exp(-0.5 * mul);
	}
	return 0;
}

int GMM::whichComponent(const Vec3d color)const {
	int tag = 0; 
	double Max = 0;
	for (int i = 0; i < ncomponent; i++) {
		double tmp = (*this)(i, color);
		if (tmp > Max)
			tag = i, Max = tmp;
	}
	return tag;
}

void GMM::initLearning() {
	for (int i = 0; i < ncomponent; i++) {
		sum[i][0] = sum[i][1] = sum[i][2] = 0;
		prod[i][0][0] = prod[i][0][1] = prod[i][0][2]
			= prod[i][1][0] = prod[i][1][1] = prod[i][1][2]
			= prod[i][2][0] = prod[i][2][1] = prod[i][2][2] = 0;
		nsample[i] = 0;
	}
	totsample = 0;
}

void GMM::addSample(int ci, const Vec3d color) {
	sum[ci][0] += color[0], sum[ci][1] += color[1], sum[ci][2] += color[2];
	prod[ci][0][0] += color[0] * color[0], prod[ci][0][1] += color[0] * color[1], prod[ci][0][2] += color[0] * color[2];
	prod[ci][1][0] += color[1] * color[0], prod[ci][1][1] += color[1] * color[1], prod[ci][1][2] += color[1] * color[2];
	prod[ci][2][0] += color[2] * color[0], prod[ci][2][1] += color[2] * color[1], prod[ci][2][2] += color[2] * color[2];
	nsample[ci]++;
	totsample++;
}

void GMM::doLearning() {
	cout << "		GMM:doLearning¿ªÊ¼\n";
	const double variance = 0.01;
	for (int i = 0; i < ncomponent; i++) {
		int n = nsample[i];
		if (n == 0)
			coefs[i] = 0;
		else {
			coefs[i] = 1.0 * n / totsample;

			double* m = mean + 3 * i;
			m[0] = sum[i][0] / n, m[1] = sum[i][1] / n, m[2] = sum[i][2] / n;
			
			double* c = cov + 9 * i;
			c[0] = prod[i][0][0] / n - m[0] * m[0], c[1] = prod[i][0][1] / n - m[0] * m[1], c[2] = prod[i][0][2] / n - m[0] * m[2];
			c[3] = prod[i][1][0] / n - m[1] * m[0], c[4] = prod[i][1][1] / n - m[1] * m[1], c[5] = prod[i][1][2] / n - m[1] * m[2];
			c[6] = prod[i][2][0] / n - m[2] * m[0], c[7] = prod[i][2][1] / n - m[2] * m[1], c[8] = prod[i][2][2] / n - m[2] * m[2];

			double det = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
			if (det <= numeric_limits<double>::epsilon()) 
				c[0] += variance, c[4] += variance, c[8] += variance;
			calcInvCovAndDet(i);
		}
	}
	cout << "		GMM:doLearning½áÊø\n";
}

void GMM::calcInvCovAndDet(int ci) {
	if (coefs[ci] > 0) {
		double* c = cov + 9 * ci;
		
		double det = covDet[ci] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);

		invCov[ci][0][0] = (c[4] * c[8] - c[5] * c[7]) / det;
		invCov[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) / det;
		invCov[ci][2][0] = (c[3] * c[7] - c[4] * c[6]) / det;
		invCov[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) / det;
		invCov[ci][1][1] = (c[0] * c[8] - c[2] * c[6]) / det;
		invCov[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) / det;
		invCov[ci][0][2] = (c[1] * c[5] - c[2] * c[4]) / det;
		invCov[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) / det;
		invCov[ci][2][2] = (c[0] * c[4] - c[1] * c[3]) / det;
	}
}