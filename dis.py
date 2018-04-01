# -*-coding:utf-8-*-
# 全局变量
INF = float('inf')
GRAPH = None


def dijkstra(src):
    # 遍历除src以外所有点
    for x in range(0, len(GRAPH)):
        if x != src:
            # 初始化并进入最小优先队列
            queue, finding, existing = [(0, [src])], set(), False
            while queue:
                # 获取当前队列中消耗最低节点和对应路径，消耗
                cost, temp_path = queue.pop(0)
                current_point = temp_path[-1]

                # 已遍历过，继续
                if current_point in finding:
                    continue

                # 到目的地，赋终值跳出
                if current_point == x:
                    distance[x], path[x], forwarding_table[x], existing = cost, temp_path[1:], temp_path[1], True
                    break

                # 将当前节点所有未访问过的相邻节点加入路径后，加入队列中，同时还加入访问该点的总消耗
                for c in range(0, len(GRAPH)):
                    if c not in finding and GRAPH[current_point][c] not in {0, INF}:
                        queue.append((cost + GRAPH[current_point][c], temp_path + [c]))

                # 队列排序，当前节点加入集合中
                queue.sort()
                finding.add(current_point)

            # 不存在路径从src到des，赋值为无限大
            if not existing:
                distance[x], path[x], forwarding_table[x] = INF, [], []


if __name__ == '__main__':
    # 赋值
    GRAPH = [[0, 7, INF, 3, 3, 2],
             [7, 0, 5, INF, 1, 2],
             [INF, 5, 0, 6, INF, 3],
             [3, INF, 6, 0, INF, 1],
             [3, 1, INF, INF, 0, INF],
             [2, 2, 3, 1, INF, 0]]

    if GRAPH:
        # 初始化
        distance = {}
        path = {}
        forwarding_table = {}

        # 实现算法打印结果
        dijkstra(3)
        print(distance)
        print(path)
        print(forwarding_table)
