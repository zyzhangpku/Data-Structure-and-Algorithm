# Assignment #6: "树"算：Huffman,BinHeap,BST,AVL,DisjointSet

Updated 1308 GMT+8 Mar 28, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763



**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)



## 1. 题目

### 22275: 二叉搜索树的遍历

http://cs101.openjudge.cn/practice/22275/



思路：二叉搜索树的建立



代码

```python
# 
class Node:
    def __init__(self, val, left=None, right=None):
        self.left = left
        self.right = right
        self.val = str(val)

    def insert(self, other):
        if int(other.val) < int(self.val):
            if self.left:
                self.left.insert(other)
            else:
                self.left = other
        elif int(other.val) > int(self.val):
            if self.right:
                self.right.insert(other)
            else:
                self.right = other

    def post_order(self):
        ans = ''
        if self.left:
            ans += self.left.post_order() + ' '
        if self.right:
            ans += self.right.post_order() + ' '
        return ans + self.val


ignore = input()
s = list(map(int, input().split()))
head = Node(s[0])
for i in s[1:]:
    head.insert(Node(i))
print(head.post_order())
```



![image-20240327144754111](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240327144754111.png)





### 05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/



思路：bfs



代码

```python
# 
class Node:
    def __init__(self, val, left=None, right=None):
        self.left = left
        self.right = right
        self.val = str(val)

    def insert(self, other):
        if int(other.val) < int(self.val):
            if self.left:
                self.left.insert(other)
            else:
                self.left = other
        elif int(other.val) > int(self.val):
            if self.right:
                self.right.insert(other)
            else:
                self.right = other


s = list(map(int, input().split()))
head = Node(s[0])
for i in s[1:]:
    head.insert(Node(i))
bfs = [head]
while bfs:
    now = bfs.pop(0)
    print(now.val, end=' ')
    if now.left:
        bfs.append(now.left)
    if now.right:
        bfs.append(now.right)

```



![image-20240326134609914](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240326134609914.png)





### 04078: 实现堆结构

http://cs101.openjudge.cn/practice/04078/

练习自己写个BinHeap。当然机考时候，如果遇到这样题目，直接import heapq。手搓栈、队列、堆、AVL等，考试前需要搓个遍。



思路：最小堆/最大堆，是用索引实现的，“树”是虚构的，删除和插入后，通过上浮下沉完成更新



代码

```python
# 
from math import floor as floor


class MinHeap:
    def __init__(self):
        self.value = []

    def get_min(self):
        if not self.value:
            return None
        return self.value[0]

    def swap(self, a, b):
        self.value[a], self.value[b] = self.value[b], self.value[a]

    def insert(self, x):
        self.value.append(x)
        index = len(self.value) - 1
        while True:
            parent_index = floor((index - 1) / 2)
            if index <= 0 or self.value[parent_index] <= self.value[index]:
                return
            self.swap(index, parent_index)
            index = parent_index

    def delete_min(self):
        if not self.value:
            return
        self.swap(0, -1)
        self.value.pop()
        index = 0
        while index < len(self.value):
            left_index, right_index = 2 * index + 1, 2 * index + 2
            if left_index >= len(self.value):
                return
            if right_index >= len(self.value):
                if self.value[index] > self.value[left_index]:
                    self.swap(index, left_index)
                    continue
                else:
                    return
            if self.value[index] < min(self.value[left_index], self.value[right_index]):
                return
            small = left_index if self.value[left_index] < self.value[right_index] else right_index
            self.swap(small, index)
            index = small


heap = MinHeap()
for i in range(int(input())):
    s = input()
    if s[0] == '1':
        heap.insert(int(s[2:]))
    else:
        print(heap.get_min())
        heap.delete_min()
```



![image-20240327191027085](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240327191027085.png)





### 22161: 哈夫曼编码树

http://cs101.openjudge.cn/practice/22161/



思路：如何构建哈夫曼树？用最小堆完成。求值则用递归

由于需要比较，需要写两个魔法函数用来比较



代码

```python
# 
import heapq


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def huff_value(self, h):
        if not self.left and not self.right:
            return h * self.val
        left_value, right_value = 0, 0
        if self.left:
            left_value = self.left.huff_value(h + 1)
        if self.right:
            right_value = self.right.huff_value(h + 1)
        return left_value + right_value

    def __lt__(self, other):
        return self.val < other.val

    def __gt__(self, other):
        return self.val > other.val


n = int(input())
nodes = []
for i in list(map(int, input().split())):
    heapq.heappush(nodes, Node(i))
while len(nodes) > 1:
    left, right = heapq.heappop(nodes), heapq.heappop(nodes)
    heapq.heappush(nodes, Node(left.val+right.val, left, right))
print(nodes[0].huff_value(0))

```



![image-20240328122310037](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240328122310037.png)





### 晴问9.5: 平衡二叉树的建立

https://sunnywhy.com/sfbj/9/5/359



思路：AVL树基本操作



代码

```python
# 
class Node:
    def __init__(self, key, height=1):
        self.key = key
        self.left = None
        self.right = None
        self.height = height

    def pre_order(self):
        ans = str(self.key)
        if self.left:
            ans += ' ' + self.left.pre_order()
        if self.right:
            ans += ' ' + self.right.pre_order()
        return ans


class AVLTree:
    def insert(self, root, key):
        if not root:
            return Node(key)
        elif key < root.key:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)

        root.height = 1 + max(self.getHeight(root.left), self.getHeight(root.right))

        balanceFactor = self.getBalance(root)

        if balanceFactor > 1:
            if key < root.left.key:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)

        if balanceFactor < -1:
            if key > root.right.key:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)

        return root

    def leftRotate(self, z):
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        return y

    def rightRotate(self, y):
        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))
        x.height = 1 + max(self.getHeight(x.left), self.getHeight(x.right))

        return x

    def getHeight(self, root):
        if not root:
            return 0
        return root.height

    def getBalance(self, root):
        if not root:
            return 0
        return self.getHeight(root.left) - self.getHeight(root.right)


n = int(input())
avl = AVLTree()
root = None
for i in list(map(int, input().split())):
    root = avl.insert(root, i)
print(root.pre_order())

```



![image-20240328130306483](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240328130306483.png)





### 02524: 宗教信仰

http://cs101.openjudge.cn/practice/02524/



思路：并查集基本操作，这里用了按秩合并优化



代码

```python
# 
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
    cnt = set([uf.find(i) for i in uf.parent])
    print(f'Case {n_case}:', len(cnt) - 1)

```



![image-20240328124130590](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240328124130590.png)





## 2. 学习总结和收获

新知识点：最大堆/最小堆、哈弗曼树、AVL树、并查集

经典的数据结构+基本操作，理解原理其实不难。

所有题目全部使用类实现，标准化接口

争取让自己的代码达到honor code的水平（虽然没加注释）

每日选做已完成90%+，剩下的几道之后两天继续做完



