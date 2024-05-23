# 目录

基础数据结构和算法

排序

线性表及其算法

链表、栈、队列

特殊算法

调度场、表达式之间的转换、单调栈

python内置库：deque

树

特殊算法

并查集、AVL、Huffman编码树、堆、字典树

图

最短路、最小生成树、拓扑排序

# 基础数据结构和算法

## 排序

### 归并排序（Merge Sort）

#### 基础知识

时间复杂度：

- **最坏情况**: *O*(*n*log*n*)
- **平均情况**: *O*(*n*log*n*)
- **最优情况**: O*(*n*log*n*)

- **空间复杂度**: O(n) — 需要额外的内存空间来存储临时数组。

- **稳定性**: 稳定 — 相同元素的相对顺序在排序后不会改变。

#### 代码示例

#### 应用

- **计算逆序对数**：在一个数组中，如果前面的元素大于后面的元素，则这两个元素构成一个逆序对。归并排序可以在排序过程中修改并计算逆序对的总数。这通过在归并过程中，每当右侧的元素先于左侧的元素被放置到结果数组时，记录左侧数组中剩余元素的数量来实现。
- **排序链表**：归并排序在链表排序中特别有用，因为它可以实现在链表中的有效排序而不需要额外的空间，这是由于链表的节点可以通过改变指针而不是实际移动节点来重新排序。

#### 例题

##### **OJ02299:Ultra-QuickSort**

http://cs101.openjudge.cn/2024sp_routine/02299/

与**20018:蚂蚁王国的越野跑**（http://cs101.openjudge.cn/2024sp_routine/20018/）类似。

算需要交换多少次来得到一个排好序的数组，其实就是算逆序对。

```python
d = 0


def merge(arr, l, m, r):
    """对l到m和m到r两段进行合并"""
    global d
    n1, n2 = m - l + 1, r - m  # L1和L2的长
    L1, L2 = arr[l:m + 1], arr[m + 1:r + 1]
    # L1和L2均为有序序列
    i, j, k = 0, 0, l  # i为L1指针，j为L2指针，k为arr指针
    '''双指针法合并序列'''
    while i < n1 and j < n2:
        if L1[i] <= L2[j]:
            arr[k] = L1[i]
            i += 1
        else:
            arr[k] = L2[j]
            d += n1 - i  # 精髓所在
            j += 1
        k += 1
    while i < n1:
        arr[k] = L1[i]
        i += 1
        k += 1
    while j < n2:
        arr[k] = L2[j]
        j += 1
        k += 1


def mergesort(arr, l, r):
    """对arr的l到r一段进行排序"""
    if l < r:  # 递归结束条件，很重要
        m = (l + r) // 2
        mergesort(arr, l, m)
        mergesort(arr, m + 1, r)
        merge(arr, l, m, r)


while True:
    n = int(input())
    if n == 0:
        break
    array = []
    for b in range(n):
        array.append(int(input()))
    d = 0
    mergesort(array, 0, n - 1)
    print(d)
```

### 快速排序（Quick Sort）

时间复杂度

- **最坏情况**: �(�2)*O*(*n*2) — 通常发生在已经排序的数组或基准选择不佳的情况下。
- **平均情况**: �(�log⁡�)*O*(*n*log*n*)
- **最优情况**: �(�log⁡�)*O*(*n*log*n*) — 适当的基准可以保证分割平衡。

- **空间复杂度**: �(log⁡�)*O*(log*n*) — 主要是递归的栈空间。

- **稳定性**: 不稳定 — 基准点的选择和划分过程可能会改变相同元素的相对顺序。

应用：k-th元素

### 堆排序（Heap Sort）

时间复杂度

- **最坏情况**: �(�log⁡�)*O*(*n*log*n*)
- **平均情况**: �(�log⁡�)*O*(*n*log*n*)
- **最优情况**: �(�log⁡�)*O*(*n*log*n*)

- **空间复杂度**: �(1)*O*(1) — 堆排序是原地排序算法，不需要额外的存储空间。

- **稳定性**: 不稳定 — 堆的维护过程可能会改变相同元素的原始相对顺序。

# 线性表

# 树



## 并查集

并查集（Union-Find 或 Disjoint Set Union，简称DSU）是一种处理不交集合的合并及查询问题的数据结构。它支持两种操作：

1. **Find**: 确定某个元素属于哪一个子集。这个操作可以用来判断两个元素是否属于同一个子集。
2. **Union**: 将两个子集合并成一个集合。

### 使用场景

并查集常用于处理一些元素分组情况，可以动态地连接和判断连接，广泛应用于网络连接、图的连通分量、最小生成树等问题。

### 核心思想

并查集通过数组或者特殊结构存储每个元素的父节点信息。初始时，每个元素的父节点是其自身，表示每个元素自成一个集合。通过路径压缩和按秩合并等优化策略，可以提高并查集的效率。

- **路径压缩**：在执行Find操作时，使得路径上的所有点直接指向根节点，这样可以减少后续操作的时间复杂度。
- **按秩合并**：在执行Union操作时，总是将较小的树连接到较大的树的根节点上，这样可以避免树过深，影响操作效率。

### 代码示例

```python
class UnionFind:
    # 初始化
    def __init__(self, size):
        # 将每个节点的上级设置为自己
        self.parent = list(range(size))
        # 每个节点的秩都是0
        self.rank = [0] * size
	
    # 查找
    def find(self, p):
        if self.parent[p] != p:
            # 这一步进行了路径压缩。
            # 如果不进行路径压缩，这一步是 return self.find(self.parent[p])
            self.parent[p] = self.find(self.parent[p])  
        return self.parent[p]
	
    # 合并
    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            # 按秩合并，总是将较小的树连接到较大的树的根节点上
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                # 如果两个节点的秩相等，就无所谓
                self.parent[rootQ] = rootP
                # 但这时需要把连接后较大的节点的秩+1
                self.rank[rootP] += 1
	
    # 是否属于同一集合
    def connected(self, p, q):
        return self.find(p) == self.find(q)
```

### 例题

#### OJ02524:宗教信仰

http://cs101.openjudge.cn/dsapre/02524/

最基本的应用，只是最后多了一步看看有多少个集合。

```python
class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size + 1)]
        self.rank = [0] * (size + 1)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_parent = self.find(x)
        y_parent = self.find(y)
        if x_parent != y_parent:
            if self.rank[x_parent] > self.rank[y_parent]:
                self.parent[y_parent] = x_parent
            elif self.rank[x_parent] < self.rank[y_parent]:
                self.parent[x_parent] = y_parent
            else:
                self.parent[y_parent] = x_parent
                self.rank[x_parent] += 1


n_case = 0
while True:
    n_case += 1
    n, m = map(int, input().split())
    if m == 0 and n == 0:
        break
    uf = UnionFind(n)
    for i in range(m):
        a, b = map(int, input().split())
        uf.union(a, b)
    cnt = set([uf.find(i) for i in uf.parent])  # 这一步是多的
    print(f'Case {n_case}:', len(cnt) - 1)
```

#### OJ18250:冰阔落 I

http://cs101.openjudge.cn/2024sp_routine/18250/

这题一开始WA，后来检查，发现原因是按秩合并时，parent[x]不一定更新了。虽然最后用self.find(x)又压缩了一次，仍然可能指向的不是最深的节点。好在此题数据小，无需按秩合并。

```python
class DJS:
    def __init__(self, size):
        self.parent = [i for i in range(size + 1)]
        self.rank = [0 for _ in range(size + 1)]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        self.parent[root_b] = root_a
        """
        if root_b != root_a:
            if self.rank[root_b] == self.rank[root_a]:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1
            elif self.rank[root_b] > self.rank[root_a]:
                self.parent[root_a] = root_b
            else:
                self.parent[root_b] = root_a
        """

    def check(self, a, b):
        if self.find(a) == self.find(b):
            print('Yes')
        else:
            print('No')


while True:
    try:
        n, m = map(int, input().split())
    except EOFError:
        break
    d = DJS(n)
    for _ in range(m):
        x, y = map(int, input().split())
        d.check(x, y)
        d.union(x, y)
    cnt = 0
    ans = []
    for i in range(1, n + 1):
        if d.find(i) == i:
            cnt += 1
            ans.append(i)
    print(len(ans))
    print(*ans)

```

#### OJ01703:发现它，抓住它

这题一开始没想出来，因为给出的条件是某两个节点属于不同的集合，而非相同的集合。但是由于一共只有两个集合，所以可以创建一个长度为2n的数组，parent[x]是和x同类的，parent[x+n]是和x不同的。

思路很新颖，值得学习。

http://cs101.openjudge.cn/2024sp_routine/01703/

```python
class DJS:
    def __init__(self, size):
        self.parent = [i for i in range(size + 1)]
        self.rank = [0 for _ in range(size + 1)]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_b != root_a:
            if self.rank[root_b] == self.rank[root_a]:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1
            elif self.rank[root_b] > self.rank[root_a]:
                self.parent[root_a] = root_b
            else:
                self.parent[root_b] = root_a

    def check(self, a, b):
        if self.find(a) == self.find(b):
            print('Yes')
        else:
            print('No')


for _ in range(int(input())):
    n, m = map(int, input().split())
    d = DJS(2 * n)
    for _ in range(m):
        info = input().split()
        a, b = map(int, info[1:])
        if info[0] == 'A':
            if d.find(a) == d.find(b) or d.find(a + n) == d.find(b + n):
                print('In the same gang.')
            elif d.find(a + n) == d.find(b) or d.find(a) == d.find(b + n):
                print('In different gangs.')
            else:
                print('Not sure yet.')
        else:
            d.union(a, b + n)
            d.union(a + n, b)
```

# 图

## Dijkstra

### 代码示例

```python
import heapq


def dijkstra(graph, start):
    # 初始化距离字典，所有顶点距离为无穷大，起始点距离为0
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    # 优先队列，用于存储每个顶点及其对应的距离，并按距离自动排序
    priority_queue = [(0, start)]

    while priority_queue:
        # 获取当前距离最小的顶点
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # 遍历当前顶点的邻接点
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            # 如果计算的距离小于已知距离，更新距离
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


# 示例图
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

# 测试算法
start_vertex = 'A'
distances = dijkstra(graph, start_vertex)
print(f"Distances from {start_vertex}: {distances}")
```

### 例题

**OJ05443:兔子与樱花**

http://cs101.openjudge.cn/dsapre/05443/

模板题目，额外的一点是需要记录路径

```python
import heapq

def dijkstra(adjacency, start):
    # 初始化，将其余所有顶点到起始点的距离都设为inf（无穷大）
    distances = {vertex: float('inf') for vertex in adjacency}
    # 初始化，所有点的前一步都是None
    previous = {vertex: None for vertex in adjacency}
    # 起点到自身的距离为0
    distances[start] = 0
    # 优先队列
    pq = [(0, start)]

    while pq:
        # 取出优先队列中，目前距离最小的
        current_distance, current_vertex = heapq.heappop(pq)
        # 剪枝，如果优先队列里保存的距离大于目前更新后的距离，则可以跳过
        if current_distance > distances[current_vertex]:
            continue

        # 对当前节点的所有邻居，如果距离更优，将他们放入优先队列中
        for neighbor, weight in adjacency[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                # 这一步用来记录每个节点的前一步
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous

def shortest_path_to(adjacency, start, end):
    # 逐步访问每个节点上一步
    distances, previous = dijkstra(adjacency, start)
    path = []
    current = end
    while previous[current] is not None:
        path.insert(0, current)
        current = previous[current]
    path.insert(0, start)
    return path, distances[end]

# Read the input data
P = int(input())
places = {input().strip() for _ in range(P)}

Q = int(input())
graph = {place: {} for place in places}
for _ in range(Q):
    src, dest, dist = input().split()
    dist = int(dist)
    graph[src][dest] = dist
    graph[dest][src] = dist  # Assuming the graph is bidirectional

R = int(input())
requests = [input().split() for _ in range(R)]

# Process each request
for start, end in requests:
    if start == end:
        print(start)
        continue

    path, total_dist = shortest_path_to(graph, start, end)
    output = ""
    for i in range(len(path) - 1):
        output += f"{path[i]}->({graph[path[i]][path[i+1]]})->"
    output += f"{end}"
    print(output)
```
