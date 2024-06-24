# Assignment #9: 图论：遍历，及 树算

Updated 1600 GMT+8 Apr 23, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763



**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)



## 1. 题目

### 04081: 树的转换

http://cs101.openjudge.cn/dsapre/04081/



思路：建树



代码

```python
# 
class Node:
    def __init__(self):
        self.children = []
        self.child = None
        self.next = None
def build(s):
    root = Node()
    stack = [root]
    depth = 0
    for q in s:
        now = stack[-1]
        if q == 'd':
            new_node = Node()
            if not now.children:
                now.child = new_node
            else:
                now.children[-1].next = new_node
            now.children.append(new_node)
            stack.append(new_node)
            depth = max(depth, len(stack) - 1)
        else:
            stack.pop()
    return root, depth
def h(node):
    if not node:
         return -1
    return max(h(node.child), h(node.next)) + 1
root, H = build(input())
print(f'{H} => {h(root)}')

```



![image-20240423195439452](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240423195439452.png)





### 08581: 扩展二叉树

http://cs101.openjudge.cn/dsapre/08581/



思路：建树，栈



代码

```python
# 
class Node:
    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right
    def order(self):
        post, mid = '', ''
        if self.left:
            post += self.left.order()[0]
            mid += self.left.order()[1]
        mid += self.key
        if self.right:
            post += self.right.order()[0]
            mid += self.right.order()[1]
        return post + self.key, mid
def build(s):
    if s == '.':
        return None
    #print(s)
    if len(s) == 3:
        return Node(s[0])
    length = len(s)
    i = 1
    stack = []
    while i < length:
        if s[i] != '.':
            stack.append(1)
        else:
            if stack and stack[-1] == 0:
                stack.pop()
                stack.pop()
            stack.append(0)
        while len(stack) >= 3 and stack[-1] == 0 and stack[-2] == 0:
            stack.pop(), stack.pop(), stack.pop()
            stack.append(0)
        #print(s[i], stack)
        if stack == [0]:
            return Node(s[0], build(s[1: i]), build(s[i+1:]))
        i += 1
tree = build(input())

#print(tree.order())
for i in tree.order()[::-1]:
    print(i)


```



![image-20240423193505663](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240423193505663.png)





### 22067: 快速堆猪

http://cs101.openjudge.cn/practice/22067/



思路：开学的时候就写过了，一种方法是堆的懒删除，一种是辅助栈。



代码

```python
# 
stack = []
aux_stack = []
while True:
    try:
        s = input().split()
        if s[0] == "pop":
            if stack:
                stack.pop()
                if aux_stack:
                    aux_stack.pop()
        elif s[0] == "min":
            if aux_stack:
                print(aux_stack[-1])
        else:
            h = int(s[1])
            stack.append(h)
            if not aux_stack:
                aux_stack.append(h)
            else:
                k = aux_stack[-1]
                aux_stack.append(min(k, h))
    except EOFError:
        break
```



![image-20240423185900564](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240423185900564.png)





### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123



思路：基本dfs



代码

```python
# 
s = 10
dx = [-2, -1, 1, 2, 2, 1, -1, -2]
dy = [1, 2, 2, 1, -1, -2, -2, -1]
ans = 0
def dfs(dep, x, y):
    if n * m == dep:
        global ans
        ans += 1
        return
    for r in range(8):
        nx = x + dx[r]
        ny = y + dy[r]
        if not chess[nx][ny] and 0 <= nx < n and 0 <= ny < m:
            chess[nx][ny] = True
            dfs(dep + 1, nx, ny)
            chess[nx][ny] = False
for _ in range(int(input())):
    n, m, x, y = map(int, input().split())
    chess = [[False] * s for _ in range(s)]
    ans = 0
    chess[x][y] = True
    dfs(1, x, y)
    print(ans)

```



![image-20240423195951094](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240423195951094.png)







## 2. 学习总结和收获

这周太忙了，只预习/复习了图论的经典算法，BF，Dijstra等，作业先做4个题，五一放假补上！





