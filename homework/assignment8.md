# Assignment #8: 图论：概念、遍历，及 树算

Updated 1600 GMT+8 Apr 9, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763



**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)





## 1. 题目

### 19943: 图的拉普拉斯矩阵

matrices, http://cs101.openjudge.cn/practice/19943/

请定义Vertex类，Graph类，然后实现



思路：Vertex类，存放每个点的编号、度数、邻接点。Graph存放所有点的实例，并且对点实现“连接”



代码

```python
# 
n, m = map(int, input().split())


class Vertex:
    def __init__(self, num, deg=0):
        self.num = num
        self.deg = deg
        self.adj = [0 for _ in range(num+1)]


class Graph:
    def __init__(self, nVertex, nEdge):
        self.nVertex = nVertex
        self.nEdge = nEdge
        self.vertexes = [Vertex(i) for i in range(nVertex)]

    def input_edges(self):
        for i in range(self.nEdge):
            a, b = map(int, input().split())
            self.connect(a, b)

    def connect(self, a, b):
        if a < b:
            a, b = b, a
        self.vertexes[a].deg += 1
        self.vertexes[b].deg += 1
        self.vertexes[a].adj[b] = 1

    def laplace(self):
        for i in range(self.nVertex):
            for j in range(self.nVertex):
                adj = self.vertexes[i].adj[j] if i > j else self.vertexes[j].adj[i]
                print((i == j) * self.vertexes[i].deg - adj, end=' ')
            print()


g = Graph(n, m)
g.input_edges()
g.laplace()

```



![image-20240410154659746](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240410154659746.png)





### 18160: 最大连通域面积

matrix/dfs similar, http://cs101.openjudge.cn/practice/18160



思路：有关图的最基本dfs。



代码

```python
# 
dx = [0, 0, 1, -1, 1, 1, -1, -1]
dy = [1, -1, 0, 0, 1, -1, 1, -1]
for _ in range(int(input())):
    a, b = map(int, input().split())
    s, now = 0, 0
    m = [input() for i in range(a)]
    visited = [[False] * b for j in range(a)]
    def dfs(x, y):
        if x < 0 or y < 0 or x >= a or y >= b or m[x][y] == '.' or visited[x][y]:
            return
        visited[x][y] = True
        global now
        now += 1
        for i in range(8):
            dfs(x + dx[i], y + dy[i])
    for x in range(a):
        for y in range(b):
            if visited[x][y] or m[x][y] == '.':
                continue
            dfs(x, y)
            s = max(s, now)
            now = 0
    print(s)
```



![image-20240409183037732](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240409183037732.png)





### sy383: 最大权值连通块

https://sunnywhy.com/sfbj/10/3/383



思路：dfs



代码

```python
# 
n, m = map(int, input().split())
w = list(map(int, input().split()))
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))
graph = [[] for _ in range(n)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)
visited = [False] * n
ans = 0


def dfs(x):
    visited[x] = True
    tot = w[x]
    for neighbor in graph[x]:
        if not visited[neighbor]:
            tot += dfs(neighbor)
    return tot


for i in range(n):
    if not visited[i]:
        ans = max(ans, dfs(i))
print(ans)

```



![image-20240416195749834](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240416195749834.png)





### 03441: 4 Values whose Sum is 0

data structure/binary search, http://cs101.openjudge.cn/practice/03441



思路：查找。暴力做是n^4的时间复杂度，a+b存一下，c+d存一下变成n^2的。但是如果用列表存，时间复杂度虽然对，但是仍然很慢。这时用字典存储（x in d.keys()复杂度是O(1)）。代价是空间占用稍大。



代码

```python
# 
a, b, c, d = [], [], [], []
q = [a, b, c, d]
n = int(input())
for i in range(n):
    s = list(map(int, input().split()))
    for j in range(4):
        q[j].append(s[j])
dict1 = {}
for i in range(n):
    for j in range(n):
        if not a[i]+b[j] in dict1:
            dict1[a[i] + b[j]] = 0
        dict1[a[i] + b[j]] += 1

cnt = 0
for i in range(n):
    for j in range(n):
        if -(c[i]+d[j]) in dict1:
            cnt += dict1[-(c[i] + d[j])]

print(cnt)

```



![image-20240416194839596](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240416194839596.png)





### 04089: 电话号码

trie, http://cs101.openjudge.cn/practice/04089/

Trie 数据结构可能需要自学下。



思路：经典的字典树



代码

```python
# 
def build_trie(s, parent: dict):
    if s[0] in parent.keys():
        if len(s) == 1:
            return False
        if not parent[s[0]]:
            return False
        return build_trie(s[1:], parent[s[0]])
    parent[s[0]] = {}
    if len(s) == 1:
        return True
    return build_trie(s[1:], parent[s[0]])
t = int(input())
for i in range(t):
    trie = {}
    n = int(input())
    flag = False
    for j in range(n):
        number = input()
        if flag:
            continue
        if not build_trie(number, trie):
            print('NO')
            flag = True
    if not flag:
        print('YES')
    #print(trie)

```



![image-20240409164935045](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240409164935045.png)





### 04082: 树的镜面映射

http://cs101.openjudge.cn/practice/04082/



思路：先根据输入建立二叉树（会用到栈）。在这个二叉树中，右节点是和自己“同级”的，左节点是自己的“下级”。这样建立二叉树之后，就能直接写出广度优先遍历，而不需要再把二叉树变为原来的树了。



代码

```python
# 
n = int(input())
seq = input().split()


class BinaryNode:
    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right
     
def build_binary_tree(s: list):
    if len(s) == 1:
        if s[0][0] == '$':
            return None
        return BinaryNode(s[0][0])
    stack = []
    for j, i in enumerate(s):
        if j == 0:
            continue
        if i[-1] == '0':
            stack.append(0)
        else:
            if stack and stack[-1] == 1:
                stack.pop()
                stack.pop()
                stack.append(1)
            else:
                stack.append(1)
        while len(stack) >= 3 and stack[-1] == 1 and stack[-2] == 1:
            for _ in range(3):
                stack.pop()
            stack.append(1)
        if len(stack) == 1 and stack[0] == 1:
            return BinaryNode(s[0][0],
                              left=build_binary_tree(s[1:j + 1]),
                              right=build_binary_tree(s[j + 1:]))


def build_tree(root: BinaryNode):
    nodes = [[root]]
    while True:
        new = []
        for node in nodes[-1]:
            if node.left:
                now = node.left
                new.append(now)
                while now.right:
                    new.append(now.right)
                    now = now.right
        if new:
            nodes.append(new)
        else:
            break
    for layer in nodes:
        for node in layer[::-1]:
            print(node.key, end=' ')


r = build_binary_tree(seq)
build_tree(r)

```



![image-20240409162609027](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240409162609027.png)





## 2. 学习总结和收获

本次题目难度适中。

学习了经典的trie。树的镜面映射，学习了如何将普通树和二叉树互相转换。

图的题目，既复习了基本概念，也练习了基本的dfs写法。

