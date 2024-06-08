#include"GcGraph.h"

void GcGraph::InitGraph() {
	cout << "		GcGraph:InitGraph开始\n";
	edge.clear();
	head.clear();
	maxflow = 0;
	s = 0, t = 0;
	nodes = 0, edges = 0;
	deep.clear();
	now.clear();
	while (!q.empty())q.pop();
	cout << "		GcGraph:InitGraph结束\n";
}

void GcGraph::AddNode() {
	head.push_back(-1);
	deep.push_back(INT_INF);
	now.push_back(-1);
	nodes++;
}

void GcGraph::AddEdge(int from, int to, double val) {
	edge.push_back({ val,to,head[from] });
	head[from] = edges++;
}

void GcGraph::AddS() {
	s = nodes;
	AddNode();
}

void GcGraph::AddT() {
	t = nodes;
	AddNode();
}

int GcGraph::getS() {
	return s;
}

int GcGraph::getT() {
	return t;
}

double GcGraph::getFlow() {
	return maxflow;
}

bool GcGraph::bfs() {
	deep.resize(nodes, INT_INF);
	while (!q.empty())q.pop();
	q.push(s); deep[s] = 0; now[s] = head[s];
	while (!q.empty()) {
		int x = q.front(); q.pop();
		for (int i = head[x]; i != -1; i = edge[i].nxt) {
			int y = edge[i].to;
			if (edge[i].val > 0 && deep[y] == INT_INF) {
				q.push(y);
				now[y] = head[y];
				deep[y] = deep[x] + 1;
				if (y == t)return 1;
			}
		}
	}
	return 0;
}

double GcGraph::dfs(int x, double flow) {
	if (x == t)return flow;
	double ans = 0;
	for (int i = now[x]; i != -1 && flow != 0; i = edge[i].nxt) {
		now[x] = i;
		int y = edge[i].to;
		if (edge[i].val > 0 && (deep[y] == deep[x] + 1)) {
			double k = dfs(y, min(flow, edge[i].val));
			if (k == 0)deep[y] = INT_INF;
			edge[i].val -= k;
			edge[i ^ 1].val += k;
			ans += k;
			flow -= k;
		}
	}
	return ans;
}

void GcGraph::maxFlow_Dinic() {
	cout << "	GCGraph:maxFlow_Dinic()开始\n";
	int cnt = 0;
	while (bfs()) {
		cnt++;
		maxflow += dfs(s, INT_INF);
	}
	cout << "		增广次数为:" << cnt << "\n";
	cout << "		maxflow为:" << maxflow << "\n";
	cout << "	GCGraph:maxFlow_Dinic()结束\n";
}

bool GcGraph::getFgTag(int ntag) {
	int etag = ntag * 2;
	return (edge[etag].val != 0);
}