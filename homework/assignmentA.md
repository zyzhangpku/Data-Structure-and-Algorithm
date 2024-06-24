# Assignment #A: 图论：算法，树算及栈

Updated 1600 GMT+8 Apr 27, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763



**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)

## 1. 题目

### 20743: 整人的提词本

http://cs101.openjudge.cn/practice/20743/



思路：简单的栈



代码

```python
# 
s = input()
stack = []
temp = ''
for i in s:
    if i == '(':
        if temp:
            stack.append(temp)
            temp = ''
        stack.append(i)
    elif i == ')':
        if temp:
            stack.append(temp)
            temp = ''
        to_add = ''
        while stack[-1] != '(':
            to_add = stack.pop() + to_add
        stack.pop()
        stack.append(to_add[::-1])
    else:
        temp += i
    #print(stack)
if temp:
    stack.append(temp)
if len(stack) != 1:
    print(''.join(stack))
else:
    print(stack[0])
```



![image-20240426224505481](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240426224505481.png)





### 02255: 重建二叉树

http://cs101.openjudge.cn/practice/02255/



思路：建树



代码

```python
# 
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def post_order(self):
        post = ''
        if self.left:
            post += self.left.post_order()
        if self.right:
            post += self.right.post_order()
        return post + self.val


def build(pre_order, in_order):
    if not pre_order or not in_order:
        return None
    #print(pre_order, in_order)
    if len(pre_order) == 1:
        return Node(pre_order[0])
    root_val = pre_order[0]
    div = 0
    while in_order[div] != root_val:
        div += 1
    left_in_order = in_order[: div]
    right_in_order = in_order[div+1:]
    div = 1
    while pre_order[div] in left_in_order:
        div += 1
        if div >= len(pre_order):
            div = len(pre_order)
            break
    return Node(root_val, left=build(pre_order[1: div], left_in_order), right=build(pre_order[div:], right_in_order))


while True:
    try:
        p, i = map(str, input().split())
        tree = build(p, i)
        print(tree.post_order())
    except EOFError:
        break
```



![image-20240426224404365](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240426224404365.png)





### 01426: Find The Multiple

http://cs101.openjudge.cn/practice/01426/

要求用bfs实现



思路：bfs



代码

```python
# 
from collections import deque
def find_multiple(n):
    q = deque()
    q.append((1 % n, "1"))
    visited = {1 % n}
    while q:
        mod, num_str = q.popleft()
        if mod == 0:
            return num_str
        for digit in '01':
            new_num_str = num_str + digit
            new_mod = (mod * 10 + int(digit)) % n
            if new_mod not in visited:
                q.append((new_mod, new_num_str))
                visited.add(new_mod)
while True:
    n = int(input())
    if n == 0:
        break
    print(find_multiple(n))

```



![image-20240427161718378](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240427161718378.png)





### 04115: 鸣人和佐助

bfs, http://cs101.openjudge.cn/practice/04115/



思路：稍复杂的bfs



代码

```python
# 
from collections import deque

m, n, tx = map(int, input().split())
g = [list(input()) for i in range(m)]
dir = [(0, 1), (1, 0), (-1, 0), (0, -1)]
start, end = None, None
for i in range(m):
    for j in range(n):
        if g[i][j] == '@':
            start = (i, j)


def bfs():
    q = deque([start + (tx, 0)])
    v = [[-1] * n for i in range(m)]
    v[start[0]][start[1]] = tx
    while q:
        x, y, t, time = q.popleft()
        time += 1
        for dx, dy in dir:
            if 0 <= x + dx < m and 0 <= y + dy < n:
                if (elem := g[x + dx][y + dy]) == '*' and t > v[x + dx][y + dy]:
                    v[x + dx][y + dy] = t
                    q.append((x + dx, y + dy, t, time))
                elif elem == '#' and t > 0 and t - 1 > v[x + dx][y + dy]:
                    v[x + dx][y + dy] = t - 1
                    q.append((x + dx, y + dy, t - 1, time))
                elif elem == '+':
                    return time
    return -1


print(bfs())

```



![image-20240427162104037](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240427162104037.png)











## 2. 学习总结和收获

这周仍然忙于期中，所以只做了四个题，现在见了很多类型的bfs，dfs，熟练很多。



