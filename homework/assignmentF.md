# Assignment #F: All-Killed 满分

Updated 1000 GMT+8 May 27, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763



**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)



## 1. 题目

### 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/



思路：建树，bfs的队列中可以多带一项“深度”



代码

```python
# 
from collections import deque
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def set_children(self, left, right):
        if left != -1:
            self.left = left
        if right != -1:
            self.right = right


n = int(input())
nodes = {v: Node(v) for v in range(1, n + 1)}
for i in range(1, n + 1):
    a, b = map(int, input().split())
    nodes[i].set_children(a, b)
bfs = deque([(nodes[1], 1)])
ans = []
max_depth = 1
last_bfs = {}
while bfs:

    cur, depth = bfs.popleft()
    max_depth = max(max_depth, depth)
    last_bfs[depth] = cur.val
    if cur.left:
        bfs.append((nodes[cur.left], depth + 1))
    if cur.right:
        bfs.append((nodes[cur.right], depth + 1))
for i in range(1, max_depth + 1):
    ans.append(last_bfs[i])
print(*ans)
```



![image-20240526145635959](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240526145635959.png)





### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/



思路：模板题



代码

```python
# 
n = int(input())
lst = list(map(int, input().split()))
ans = []
stack = []
for i in range(n, 0, -1):
    while stack and lst[stack[-1]] <= lst[i]:
        stack.pop()
    if not stack:
        ans.append(0)
    else:
        ans.append(stack[-1])
    stack.append(i)
for i in range(n, 0, -1):
    print(ans[i], end=' ')

```



![image-20240526152249332](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240526152249332.png)





### 09202: 舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/



思路：拓扑排序（Kahn算法）用来检测有向图是否有环



代码

```python
# 
from collections import deque
def topo(g, deg, v):
    q = deque([x for x in deg.keys() if deg[x] == 0])
    cnt = 0
    while q:
        now = q.popleft()
        cnt += 1
        for next in g[now]:
            deg[next] -= 1
            if deg[next] == 0:
                q.append(next)
    if cnt == v:
        print('No')
    else:
        print('Yes')
for _ in range(int(input())):
    v, m = map(int, input().split())
    g = {a: [] for a in range(1, v + 1)}
    deg = {a: 0 for a in range(1, v + 1)}
    for _ in range(m):
        x, y = map(int, input().split())
        deg[y] += 1
        g[x].append(y)
    topo(g, deg, v)
```



![image-20240526161522462](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240526161522462.png)





### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



思路：二分，之前见到的这种题较少



代码

```python
# 
n, m = map(int, input().split())
L = list(int(input()) for x in range(n))
def check(x):
    num, cut = 1, 0
    for i in range(n):
        if cut + L[i] > x:
            num += 1
            cut = L[i]  
        else:
            cut += L[i]
    return num <= m
maxmax = sum(L)
minmax = max(L)
while minmax < maxmax:
    middle = (maxmax + minmax) // 2
    if check(middle): 
        maxmax = middle
    else:
        minmax = middle + 1  
print(maxmax)
```



![image-20240526163040332](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240526163040332.png)





### 07735: 道路

http://cs101.openjudge.cn/practice/07735/



思路：dijkstra，但是有点区别，加入优先队列的条件不是距离更短，而是金币够用，但是优先队列的比较仍然是用距离比的



代码

```python
# 
import heapq

k, n, r = int(input()), int(input()), int(input())


def dij(g, s, e):
    dis = {v: float('inf') for v in range(1, n + 1)}
    dis[s] = 0
    q = [(0, s, 0)]
    heapq.heapify(q)
    while q:
        d, now, fee = heapq.heappop(q)
        if now == n:
            return d
        for neighbor, distance, c in g[now]:
            if fee + c <= k:
                dis[neighbor] = distance + d
                heapq.heappush(q, (distance + d, neighbor, fee + c))
    return -1


g = {v: [] for v in range(1, n + 1)}
for _ in range(r):
    s, e, m, j = map(int, input().split())
    g[s].append((e, m, j))
p = dij(g, 1, n)
print(p)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



思路：并查集，和“发现他，抓住他”很像



代码

```python
# 
n, k = map(int, input().split())
cnt = 0
ds = []  # 本身, 被x吃, 吃x

for i in range(3 * n + 1):
    ds.append(i)


def find(a):
    # print(a)
    if ds[a] != a:
        ds[a] = find(ds[a])
    return ds[a]


def union(a, b):
    root_a, root_b = find(a), find(b)
    ds[root_a] = root_b


def check(d, a, b):
    if d == 1:
        return find(a + n) == find(b) or find(b + n) == find(a)
    else:
        return find(a) == find(b) or find(b) == find(a + 2 * n)


for _ in range(k):
    d, x, y = map(int, input().split())
    if x > n or y > n or check(d, x, y):
        cnt += 1
        continue
    if d == 1:
        for i in range(3):
            union(x + i * n, y + i * n)
    elif d == 2:
        union(y, x + n)
        union(x, y + 2 * n)
        union(x + 2 * n, y + n)
print(cnt)

```



![image-20240527094221314](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240527094221314.png)





## 2. 学习总结和收获

数算知识点总结：https://github.com/zyzhangpku/Data-Structure-and-Algorithm/blob/main/notes.md

（持续更新）

道路这题有点难，对dijkstra算法的掌握不是很熟练，但是题解很容易能看明白，差一点灵活运用的熟练度上

食物链和发现它抓住它非常像，这学期遇到了并查集，要么就是单纯的查找，要么就是像这两题一样，维护长度2n或者3n的数组。在以上的总结已经把同类的知识点的题放到一起了，这样可以看到，每个考点出的题，只是在模板的基础上加了点东西和技巧而已。

拓扑排序自然不用多说，月度开销的二分很有意思，学习到了，树的题目就比较简单。



