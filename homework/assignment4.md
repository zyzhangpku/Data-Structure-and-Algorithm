# Assignment #4: 排序、栈、队列和树

Updated 1600 GMT+8 Mar 12, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763



**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)



## 1. 题目

### 05902: 双端队列

http://cs101.openjudge.cn/practice/05902/



思路：双端队列，用python的deque实现比直接list实现，时间复杂度更低

手搓双端队列的话就双指针



代码

```python
# 
from collections import deque

for _ in range(int(input())):
    n = int(input())
    q = deque([])
    for i in range(n):
        a, b = map(int, input().split())
        if a == 1:
            q.append(b)
        else:
            if b == 0:
                q.popleft()
            else:
                q.pop()
    if q:
        for i in q:
            print(i, end=' ')
        print()
    else:
        print('NULL')
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240312165403839](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240312165403839.png)



### 02694: 波兰表达式

http://cs101.openjudge.cn/practice/02694/



思路：简单的栈，递归



代码

```python
# 
ops = ['+', '-', '*', '/']
s = input().split()
for i in range(len(s)):
    if s[i] not in ops:
        s[i] = float(s[i])
stack = []


def check():

    if len(stack) < 3:
        return
    if stack[-1] not in ops and stack[-2] not in ops and stack[-3] in ops:
        a, b = stack.pop(), stack.pop()
        op = stack.pop()
        if op == '+':
            ans = a + b
        elif op == '-':
            ans = b - a
        elif op == '*':
            ans = b * a
        else:
            ans = b / a
        stack.append(ans)
        check()


for i in s:
    stack.append(i)
    if type(i) != 0:
        check()
#print(stack)
print("%.6f"%stack[0])
```



![image-20240312140859464](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240312140859464.png)





### 24591: 中序表达式转后序表达式

http://cs101.openjudge.cn/practice/24591/



思路：

栈的经典题目，算法是Shunting Yard Algorithm，两个栈（实际上是一个），一个运算符栈，一个输出栈（这个不用栈保存，直接输出也可）

但是用二叉树来说更好理解，叶节点是数字，非叶节点是运算符，需要把一个中序遍历转换为后序遍历

一开始想把表达式建成二叉树，但是有点麻烦，而且之前在27637:括号嵌套二叉树等题目中练过了这种递归建树，就试着用栈写

Shunting Yard Algorithm的理解：

从左到右遍历中序表达式，

若遇到数字，直接加到输出栈，因为后序遍历中，左右叶节点是最先的

若遇到左括号，加入运算符栈，因为左括号是要建立单独的树，是运算符优先级的一种区分

若遇到右括号，从最后一个开始，将运算符中的东西弹出并加入输出栈，直到遇到左括号，因为这代表这一整个子树的建立。

若遇到运算符，则碰到了非叶节点，弹出栈中任何优先级比当前运算符更高或与当前运算符相等（优先级更高或相等代表子树深度小，所以先输出）的运算符，并将它们添加到输出队列中，然后将自己添加到运算符栈，等待右子树

代码

```python
# 
operators = ['+', '-', '*', '/']


def is_num(s):
    for i in operators + ['(', ')']:
        if i in s:
            return False
    return True


def process(raw_input):
    # convert the raw input into separated sequence
    temp, ans = '', []
    for i in raw_input.strip():
        if is_num(i):
            temp += i
        else:
            if temp:
                ans.append(temp)
            ans.append(i)
            temp = ''
    if temp:
        ans.append(temp)
    return ans


def infix_to_postfix(expression):
    # Shunting Yard Algorithm
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output_stack, op_stack = [], []
    for i in expression:
        if is_num(i):
            output_stack.append(i)
        elif i == '(':
            op_stack.append(i)
        elif i == ')':
            while op_stack[-1] != '(':
                output_stack.append(op_stack.pop())
            op_stack.pop()
        else:
            while op_stack and op_stack[-1] in operators and precedence[i] <= precedence[op_stack[-1]]:
                output_stack.append(op_stack.pop())
            op_stack.append(i)
    if op_stack:
        output_stack += op_stack[::-1]
    return output_stack


n = int(input())
for i in range(n):
    tokenized = process(input())
    print(' '.join(infix_to_postfix(tokenized)))

```



![image-20240312162035377](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240312162035377.png)





### 22068: 合法出栈序列

http://cs101.openjudge.cn/practice/22068/



思路：

栈+模拟

代码

```python
# 
def legal(origin, seq):
    if len(seq) != len(origin):
        return False
    if len(seq) == 1:
        return origin == seq
    for i in seq:
        if i not in origin:
            return False
    for i in origin:
        if i not in seq:
            return False
    stack, bank = [], list(origin)
    for i in seq:
        if bank and not stack:
            stack.append(bank.pop(0))
        while bank and stack[-1] != i:
            stack.append(bank.pop(0))
        if stack.pop() != i:
            return False
    return True


s = input()
while True:
    try:
        d = input()
    except EOFError:
        break
    if legal(s, d):
        print('YES')
    else:
        print('NO')

```



![image-20240313163113008](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240313163113008.png)





### 06646: 二叉树的深度

http://cs101.openjudge.cn/practice/06646/



思路：二叉树基本操作



代码（这个题的代码比较乱是因为这题是之前做数算pre的时候先做完27638:求二叉树的高度和叶子数目的时候修改的，这俩题是一个知识点但是有点细节的差别，27638的AC Code放在下面）

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
nodes = [Node(num=i) for i in range(n+1)]
for i in range(n):
    l, r = map(int, input().split())
    now = nodes[i+1]
    now.num = i+1
    if l != -1:
        now.left = nodes[l]
    if r != -1:
        now.right = nodes[r]
max_height, leaves = 0, 0
for i in nodes[1:]:
    max_height = max(max_height, i.cal_height())
print(max_height+1)
```

27638:求二叉树的高度和叶子数 AC Code:

```python
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



![image-20240312140914957](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240312140914957.png)





### 02299: Ultra-QuickSort

http://cs101.openjudge.cn/practice/02299/



思路：

归并排序，中间的一步有略微改动版本

代码

```python
# 
def merge_sort(lst):
    if len(lst) <= 1:
        return lst, 0
    middle = len(lst) // 2
    left, inv_left = merge_sort(lst[:middle])
    right, inv_right = merge_sort(lst[middle:])
    merged, inv_merge = merge(left, right)
    return merged, inv_left + inv_right + inv_merge


def merge(left, right):
    merged = []
    inv_count = 0
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            inv_count += len(left) - i
    merged += left[i:] + right[j:]
    return merged, inv_count


while True:
    n = int(input())
    if n == 0:
        break
    data = [int(input()) for i in range(n)]
    _, inversions = merge_sort(data)
    print(inversions)
```



![image-20240313163236518](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240313163236518.png)





## 2. 学习总结和收获

正在建设我的github[数算学习页面](https://github.com/zyzhangpku/Data-Structure-and-Algorithm/blob/PKU-DSA-(B)/Starred%20Problems.md)，不定期更新题目和知识点，但是目前内容较少，之后会加一些题目

跟进每日选做，但是不能满足与这些数据结构的基本操作，需要掌握它们在具体题目中的应用，如何选择合适的数据结构对问题进行建模，并且将实际题目的理解映射到对应数据结构的操作中去，这是需要更多训练的。
