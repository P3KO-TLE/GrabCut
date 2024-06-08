#include"common.h"

class GcGraph {
	//节点索引与节点坐标的关系:id=r*w+c
private:
	//INT型无穷大
	const int INT_INF = 1e9;
	//节点结构体
	/*struct gnode {
		double gray;
		int tag;
	};*/
	//单向边结构体
	struct gedge {
		double val;
		int to;
		int nxt;
	};
	//节点实例
	//vector<gnode> node;
	//单向边实例
	vector<gedge> edge;
	//邻接链表头
	vector<int> head;
	//最大流
	double maxflow;
	//源点和汇点
	int s, t;
	//节点数量，单向边数量
	int nodes, edges;
	//dinic算法相关的数据结构
	vector<int>deep;
	vector<int>now;
	queue<int>q;
public:
	//初始化成员变量
	void InitGraph();
	//添加节点
	void AddNode();
	//添加单向边
	void AddEdge(int from, int to, double val);
	//添加源点
	void AddS();
	//添加汇点
	void AddT();
	//获取源点索引
	int getS();
	//获取汇点索引
	int getT();
	//获取最大流
	double getFlow();
	//Dinic算法计算网络流
	void maxFlow_Dinic();
	//通过节点索引获取标签
	bool getFgTag(int ntag);
private:
	//Dinic算法相关函数
	bool bfs();
	double dfs(int x, double flow);
};