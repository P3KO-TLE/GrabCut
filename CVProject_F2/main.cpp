#include"GCparams.h"
#include"GcGraph.h"

const string filePath = "D:/CV/Resources/GC2.jpg";
const string winName = "myWin";
Mat src;
Rect rect;
GcParam params;
GcGraph graph;

void buildGraph(const Mat& src, GcParam& params, GcGraph& graph) {
	cout << "	main:buildGraph开始\n";
	int cols = src.cols, rows = src.rows;
	//初始化
	graph.InitGraph();
	//建点
	cout << "		buildGraph:建点开始\n";
	for (int y = 0; y < rows; y++)for (int x = 0; x < cols; x++)
		graph.AddNode();
	graph.AddS();
	graph.AddT();
	cout << "		buildGraph:建点结束\n";
	//连边
	//源点向像素点连边
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
	//像素点向汇点连边
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
	//象素点间相互连边
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
	cout << "	main:buildGraph结束\n";
}

void doCut(const Mat& src, GcParam& params, GcGraph& graph) {
	cout << "	main:doCut结束\n";
	graph.maxFlow_Dinic();
	for (int y = 0; y < src.rows; y++)for (int x = 0; x < src.cols; x++) {
		if (params.getFBTag(y, x) >= 2) {
			if (graph.getFgTag(y * src.cols + x))
				params.setFgd(y, x);
			else
				params.setBgd(y, x);
		}
	}
	cout << "	main:doCut结束\n";
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
	//参数模型初始化
	cout << "main:参数模型初始化开始\n";
	params.init(&src);
	params.calcBeta();
	params.calcNCost();
	cout << "main:参数模型初始化结束\n";
	//添加响应事件
	cout << "main:获取rect开始\n";
	namedWindow(winName, 0);
	resizeWindow(winName, src.cols, src.rows);
	setMouseCallback(winName, on_mouse, 0);
	imshow(winName, src);
	waitKey();
	cout << "	rect:x:"<<rect.x<<" y:"<<rect.y<<" width:"<<rect.width<<" height:"<<rect.height<<"\n";
	cout << "main:获取rect结束\n";
	//初始化参数模型的混合高斯模型
	cout << "main:初始化参数模型的混合高斯模型开始\n";
	params.initMask(rect);
	params.initGMM();
	cout << "main:初始化参数模型的混合高斯模型结束\n";
	//迭代优化
	cout << "main:迭代优化开始\n";
	for (int i = 0; i < itercount; i++) {
		params.assignGMMComponents();
		params.learnGMM();
		buildGraph(src, params, graph);
		doCut(src, params, graph);
	}
	cout << "main:迭代优化结束\n";
	//展示结果
	cout << "main:展示结果开始\n";
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
	cout << "main:展示结果结束\n";
}

int main() {
	src = imread(filePath, 1);
	grabcut(src, winName, params, graph, 20);

	system("pause");
	return 0;
}