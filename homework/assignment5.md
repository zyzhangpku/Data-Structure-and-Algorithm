# Assignment #5: "树"算：概念、表示、解析、遍历

Updated 1600 GMT+8 Mar 12, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763



**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)

## 1. 题目

### 27638: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/practice/27638/



思路：树的简单操作



代码

```python
# 
class Node:
    def __init__(self, parent=None, left=-1, right=-1, num=0):
        self.parent = parent
        self.left = nodes[left] if left != -1 else None
        self.right = nodes[right] if right != -1 else None
        self.num = num
        self.height = None

    def cal_height(self):
        if self.height:
            return self.height
        if not self.left and not self.right:
            self.height = 0
        else:
            x = 0
            if self.left:
                x = max(x, self.left.cal_height())
            if self.right:
                x = max(x, self.right.cal_height())
            self.height = x + 1
        return self.height


n = int(input())
nodes = [Node(num=i) for i in range(n)]
for i in range(n):
    l, r = map(int, input().split())
    now = nodes[i]
    now.num = i
    if l != -1:
        now.left = nodes[l]
    if r != -1:
        now.right = nodes[r]
max_height, leaves = 0, 0
for i in nodes:
    max_height = max(max_height, i.cal_height())
    if i.height == 0:
        #print(i.num)
        leaves += 1
print(max_height, leaves)
```



![image-20240322195131485](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240322195131485.png)





### 24729: 括号嵌套树

http://cs101.openjudge.cn/practice/24729/



思路：基本操作



代码

```python
# 
class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

    def order(self):
        pre, post = self.name, ''
        for child in self.children:
            pre += child.order()[0]
            post += child.order()[1]
        return pre, post + self.name


def build_tree(s):
    #print(s)
    if len(s) == 1:
        return Node(s)
    children = s[2:-1] + ','
    root = Node(s[0])
    cnt = 0
    cut = 0
    for i in range(len(children)):
        if children[i] == '(':
            cnt += 1
        elif children[i] == ')':
            cnt -= 1
        elif children[i] == ',':
            if cnt == 0:
                root.children.append(build_tree(children[cut:i]))
                cut = i + 1
    return root
root = build_tree(input())
print(root.order()[0])
print(root.order()[-1])
```



![image-20240322203606147](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240322203606147.png)





### 02775: 文件结构“图”

http://cs101.openjudge.cn/practice/02775/



思路：递归，建树。file是叶，前序遍历



代码

```python
# 
prefix = '|     '
flag = False


class filepath:
    def __init__(self, name='ROOT'):
        self.child_path = []
        self.files = []
        self.name = name

    def show_files(self, pre=''):
        print(pre+self.name)
        self.files.sort()
        for f in self.child_path:
            f.show_files(pre+prefix)
        for f in self.files:
            print(pre+f)


def build_path(now_path):
    while True:
        x = input()
        if x[0] == 'f':
            now_path.files.append(x)
        elif x[0] == 'd':
            new_path = filepath(x)
            now_path.child_path.append(new_path)
            build_path(new_path)
        else:
            if x == '#':
                global flag
                flag = True
            return


n = 0
while True:
    n += 1
    root = filepath()
    build_path(root)
    if flag:
        break
    print(f'DATA SET {n}:')
    root.show_files()
    print()
```



![image-20240322195026599](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240322195026599.png)





### 25140: 根据后序表达式建立队列表达式

http://cs101.openjudge.cn/practice/25140/



思路：按层次遍历表达式树的结果前后颠倒就得到队列表达式，这句提示是关键



代码

```python
# 
class Node:
    def __init__(self, name, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right


def build(s):
    stack = []
    for i in s:
        if ord(i) > ord('Z'):
            stack.append(Node(i))
        else:
            r, l = stack.pop(), stack.pop()
            stack.append(Node(i, l, r))
    return stack[0]


for _ in range(int(input())):
    s = input()
    tree = build(s)
    bfs = [tree]
    ans = ''
    while bfs:
        now = bfs.pop(0)
        ans += now.name
        if now.left:
            bfs.append(now.left)
        if now.right:
            bfs.append(now.right)
    print(ans[::-1])

```



![image-20240322211049654](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240322211049654.png)





### 24750: 根据二叉树中后序序列建树

http://cs101.openjudge.cn/practice/24750/



思路：跟下题一样



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
        p, i = input(), input()
        tree = build(p, i)
        print(tree.post_order())
    except EOFError:
        break
```



![image-20240322211655278](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240322211655278.png)





### 22158: 根据二叉树前中序序列建树

http://cs101.openjudge.cn/practice/22158/



思路：之前在数算pre每日选做里面做过了，建树



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
        p, i = input(), input()
        tree = build(p, i)
        print(tree.post_order())
    except EOFError:
        break

```



![image-20240322201916298](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240322201916298.png)





## 2. 学习总结和收获

这次作业较简单，都是基本的建树

对树的理解更深了，前序、后序、中序，他们的转换和栈是紧密相连的！





