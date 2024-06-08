#include"GCparams.h"
#include"GcGraph.h"

const string filePath = "D:/CV/Resources/GC2.jpg";
const string winName = "myWin";
Mat src;
Rect rect;
GcParam params;
GcGraph graph;

void buildGraph(const Mat& src, GcParam& params, GcGraph& graph) {
	cout << "	main:buildGraph��ʼ\n";
	int cols = src.cols, rows = src.rows;
	//��ʼ��
	graph.InitGraph();
	//����
	cout << "		buildGraph:���㿪ʼ\n";
	for (int y = 0; y < rows; y++)for (int x = 0; x < cols; x++)
		graph.AddNode();
	graph.AddS();
	graph.AddT();
	cout << "		buildGraph:�������\n";
	//����
	//Դ�������ص�����
	int s = graph.getS(), t = graph.getT();
	double lambda = 1000;
	for (int y = 0; y < rows; y++)for (int x = 0; x < cols; x++) {
		int id = y * cols + x;
		int tag = params.getFBTag(y, x);
		switch (tag)
		{
		case 0:
			graph.AddEdge(s, id, 0);
			graph.AddEdge(id, s, 0);
			break;
		case 1:
			graph.AddEdge(s, id, lambda);
			graph.AddEdge(id, s, 0);
		default:
			graph.AddEdge(s, id, params.getTCost(src.at<Vec3b>(y, x), 0));
			graph.AddEdge(id, s, 0);
			break;
		}
	}
	//���ص���������
	for (int y = 0; y < rows; y++)for (int x = 0; x < cols; x++) {
		int id = y * cols + x;
		int tag = params.getFBTag(y, x);
		switch (tag)
		{
		case 0:
			graph.AddEdge(id, t, lambda);
			graph.AddEdge(t, id, 0);
			break;
		case 1:
			graph.AddEdge(id, t, 0);
			graph.AddEdge(t, id, 0);
		default:
			graph.AddEdge(id, t, params.getTCost(src.at<Vec3b>(y, x), 1));
			graph.AddEdge(t, id, 0);
			break;
		}
	}
	//���ص���໥����
	for (int y = 0; y < rows; y++)for (int x = 0; x < cols; x++) {
		int id = y * cols + x;
		double cost;
		if (x > 0) {
			cost = params.getNCost(y, x, 0);
			graph.AddEdge(id, id - 1, cost);
			graph.AddEdge(id - 1, id, cost);
		}
		if (x > 0 && y > 0) {
			cost = params.getNCost(y, x, 1);
			graph.AddEdge(id, id - cols - 1, cost);
			graph.AddEdge(id - cols - 1, id, cost);
		}
		if (y > 0) {
			cost = params.getNCost(y, x, 2);
			graph.AddEdge(id, id - cols, cost);
			graph.AddEdge(id - cols, id, cost);
		}
		if (y > 0 && x < cols - 1) {
			cost = params.getNCost(y, x, 3);
			graph.AddEdge(id, id - cols + 1, cost);
			graph.AddEdge(id - cols + 1, id, cost);
		}
	}
	cout << "	main:buildGraph����\n";
}

void doCut(const Mat& src, GcParam& params, GcGraph& graph) {
	cout << "	main:doCut����\n";
	graph.maxFlow_Dinic();
	for (int y = 0; y < src.rows; y++)for (int x = 0; x < src.cols; x++) {
		if (params.getFBTag(y, x) >= 2) {
			if (graph.getFgTag(y * src.cols + x))
				params.setFgd(y, x);
			else
				params.setBgd(y, x);
		}
	}
	cout << "	main:doCut����\n";
}

static void on_mouse(int event, int x, int y, int flag, void* param) {
	switch (event) {
	case EVENT_LBUTTONDOWN:
		rect.x = x, rect.y = y;
		break;
	case EVENT_RBUTTONDOWN:
		rect.width = x - rect.x, rect.height = y - rect.y;
	default:
		break;
	}
}

void grabcut(const Mat& src, const string& winName, GcParam& params, GcGraph& graph, const int& itercount) {
	//����ģ�ͳ�ʼ��
	cout << "main:����ģ�ͳ�ʼ����ʼ\n";
	params.init(&src);
	params.calcBeta();
	params.calcNCost();
	cout << "main:����ģ�ͳ�ʼ������\n";
	//�����Ӧ�¼�
	cout << "main:��ȡrect��ʼ\n";
	namedWindow(winName, 0);
	resizeWindow(winName, src.cols, src.rows);
	setMouseCallback(winName, on_mouse, 0);
	imshow(winName, src);
	waitKey();
	cout << "	rect:x:"<<rect.x<<" y:"<<rect.y<<" width:"<<rect.width<<" height:"<<rect.height<<"\n";
	cout << "main:��ȡrect����\n";
	//��ʼ������ģ�͵Ļ�ϸ�˹ģ��
	cout << "main:��ʼ������ģ�͵Ļ�ϸ�˹ģ�Ϳ�ʼ\n";
	params.initMask(rect);
	params.initGMM();
	cout << "main:��ʼ������ģ�͵Ļ�ϸ�˹ģ�ͽ���\n";
	//�����Ż�
	cout << "main:�����Ż���ʼ\n";
	for (int i = 0; i < itercount; i++) {
		params.assignGMMComponents();
		params.learnGMM();
		buildGraph(src, params, graph);
		doCut(src, params, graph);
	}
	cout << "main:�����Ż�����\n";
	//չʾ���
	cout << "main:չʾ�����ʼ\n";
	Mat imgshow(src.rows, src.cols, CV_8UC1);
	imgshow.setTo(Scalar(0));
	for (int y = 0; y < src.rows; y++)for (int x = 0; x < src.cols; x++) {
		int tag = params.getFBTag(y, x);
		if (tag % 2 == 1)
			imgshow.at<uchar>(y, x) = 255;
	}
	namedWindow("imgshow", 0);
	resizeWindow("imgshow", imgshow.cols, imgshow.rows);
	imshow("imgshow", imgshow);
	waitKey();
	cout << "main:չʾ�������\n";
}

int main() {
	src = imread(filePath, 1);
	grabcut(src, winName, params, graph, 20);

	system("pause");
	return 0;
}