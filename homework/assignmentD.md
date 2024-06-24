# Assignment #D: May月考

Updated 1400 GMT+8 May 20, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763



**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)



## 1. 题目

### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



思路：模拟



代码

```python
# 
L, m = map(int, input().split())

dp = [1]*(L+1)

for i in range(m):
    s, e = map(int, input().split())
    for j in range(s, e+1):
        dp[j] = 0

print(dp.count(1))
```



![image-20240515094036622](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240515094036622.png)





### 20449: 是否被5整除

http://cs101.openjudge.cn/practice/20449/



思路：math



代码

```python
# 
s = input()
n = 0
for i in s:
    n =n * 2+int(i)
    if n%5==0:
        print(1,end='')
    else:
        print(0,end='')
```



![image-20240515094107717](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240515094107717.png)





### 01258: Agri-Net

http://cs101.openjudge.cn/practice/01258/



思路：prim模板



代码

```python
# 
import heapq

while True:
    try:
        n = int(input())
    except EOFError:
        break
    visited = {0}
    m_cost = 0
    g = [list(map(int, input().split())) for _ in range(n)]
    edges = [(cost, 0, to) for to, cost in enumerate(g[0])]
    heapq.heapify(edges)
    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            m_cost += cost
            for neighbor, next_cost in enumerate(g[to]):
                if neighbor not in visited:
                    heapq.heappush(edges, (next_cost, to, neighbor))
    print(m_cost)

```



![image-20240520132905419](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240520132905419.png)





### 27635: 判断无向图是否连通有无回路(同23163)

http://cs101.openjudge.cn/practice/27635/



思路：dfs



代码

```python
# 
v, e = map(int, input().split())
g = [[0 for _ in range(v)] for _ in range(v)]
ans = [True, False]
for i in range(e):
    a, b = map(int, input().split())
    if a == b:
        ans[1] = True
    g[a][b] = 1
    g[b][a] = 1
visited = [False for _ in range(v)]


def dfs(x, father):
    # print(x, father)
    for adj in range(v):
        if g[x][adj]:
            if visited[adj]:
                if adj != father and father != -1:
                    #print(x, father, 'sb')
                    ans[1] = True
                continue
            visited[adj] = True
            dfs(adj, x)


visited[0] = True
dfs(0, -1)
# print(visited)
for c in visited:
    ans[0] = ans[0] and c
for x in visited:
    if not visited[x]:
        dfs(x, -1)
if ans[0]:
    print('connected:yes')
else:
    print('connected:no')
if ans[1]:
    print('loop:yes')
else:
    print('loop:no')

```



![image-20240520125851544](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240520125851544.png)







### 27947: 动态中位数

http://cs101.openjudge.cn/practice/27947/



思路：两个堆的思想很有意思，用大跟堆存储小的数，小根堆存储大的数



代码

```python
# 
import heapq

n = int(input())
for _ in range(n):
    m = list(map(int, input().split()))
    ans = []
    big, small = [], []
    heapq.heapify(big)
    heapq.heapify(small)
    cnt = 0
    for i in m:
        cnt += 1
        if not big and not small or big[0] <= i:
            heapq.heappush(big, i)
        else:
            heapq.heappush(small, -i)
        if len(big) - len(small) >= 2:
            heapq.heappush(small, -heapq.heappop(big))
        elif len(small) > len(big):
            heapq.heappush(big, -heapq.heappop(small))
        if cnt % 2:
            ans.append(str(big[0]))
    print(len(ans))
    print(' '.join(ans))



```



![image-20240520135648911](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240520135648911.png)





### 28190: 奶牛排队

http://cs101.openjudge.cn/practice/28190/



思路：单调栈，结合题解复习



代码

```python
# 
from bisect import bisect_right as bl

lis, q1, q2, ans = [int(input()) for _ in range(int(input()))], [-1], [-1], 0
for i in range(len(lis)):
    while len(q1) > 1 and lis[q1[-1]] >= lis[i]: q1.pop()
    while len(q2) > 1 and lis[q2[-1]] < lis[i]: q2.pop()
    id = bl(q1, q2[-1])
    if id < len(q1): ans = max(ans, i - q1[id] + 1)
    q1.append(i)
    q2.append(i)
print(ans)

```



![image-20240520135853598](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240520135853598.png)





## 2. 学习总结和收获

2计概+2模板+2灵活运用，模板题问题不大，最重要是提速。灵活运用的题，多数是多个数据结构或算法的结合，一方面需要对每个知识点灵活掌握，另一方面需要经常总结遇到的题。





