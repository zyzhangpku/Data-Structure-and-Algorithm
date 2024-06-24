# Assignment #7: April 月考

Updated 1232 GMT+8 Apr 4, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763



**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)



## 1. 题目

### 27706: 逐词倒放

http://cs101.openjudge.cn/practice/27706/



思路：[::-1]切片反转



代码

```python
# 
s = input().split()[::-1]
for i in s:
    print(i, end=' ')
```



![image-20240404123325129](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240404123325129.png)





### 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/



思路：队列。由于数据量小导致pop（0）也能过。优化自然是deque



代码

```python
# 
m, n = map(int, input().split())
cnt = 0
seq = list(map(int, input().split()))
q = []
for i in seq:
    if i not in q:
        cnt += 1
        if len(q) == m:
            q.pop(0)
        q.append(i)
    else:
        pass
print(cnt)
```



![image-20240404123422742](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240404123422742.png)





### 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/



思路：很简单但是考试时确实被卡了几分钟。这提醒我们要注意边际极端情况的处理



代码

```python
# 
n, k = map(int, input().split())
seq = sorted(list(map(int, input().split())))
if k == 0:
    if seq[0] - 1 <= 0:
        print(-1)
    else:
        print(seq[0]-1)
else:
    if n == k:
        print(seq[-1])
    elif seq[k - 1] == seq[k]:
        print(-1)
    else:
        print(seq[k-1])
```



![image-20240404123526923](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240404123526923.png)





### 27948: FBI树

http://cs101.openjudge.cn/practice/27948/



思路：递归建树，老生常谈



代码

```python
# 
n = int(input())
s = input()
class Node:
    def __init__(self, v, left=None, right=None):
        self.v = v
        self.left = left
        self.right = right
    def post(self):
        ans = ''
        if self.left:
            ans += self.left.post()
        if self.right:
            ans += self.right.post()
        return ans + self.v
def f_b_i(x):
    sum = 0
    for i in x:
        sum += int(i)
    if sum == 0:
        return 'B'
    if sum == len(x):
        return 'I'
    return 'F'
def build(x):
    if len(x) == 1:
        return Node(f_b_i(x))
    return Node(f_b_i(x), left=build(x[:len(x)//2]), right=build(x[len(x)//2:]))
print(build(s).post())
```



![image-20240404123623685](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240404123623685.png)





### 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/



思路：最优做法是多队列。我在考场上直接暴力了。但是暴力解其实就是计概c水平，多队列才是数算思维



代码

```python
# 
t = int(input())
pointers = [-1] * t
q = []
head = 0
id_to_group = {}
for i in range(t):
    for id in input().split():
        id_to_group[id] = i
while True:
    command = input()
    if command == 'STOP':
        break
    if command[0] == 'E':
        id = command.split()[-1]
        group = id_to_group[id]
        if pointers[group] == -1:
            pointers[group] = len(q)
        else:
            base = pointers[group]
            for i in range(len(pointers)):
                if pointers[i] >= base:
                    pointers[i] += 1
            #pointers[group] += 1
        q.insert(pointers[group], id)
        #print(q)
        #print(pointers)
    else:
        print(q[head])
        q.pop(0)
```



![image-20240404125652341](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240404125652341.png)





### 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/



思路：简单的递归而已，但是考试时读题读了好久，对我来说，题目表述得有些模糊，应该指出给出的是所有节点之间的父子关系，我花了好长时间才看懂样例，但是题目本身没毛病也不难



代码

```python
# 
class Node:
    def __init__(self, key, children=None):
        self.key = key
        self.children = children if children is not None else []
    def order(self):
        if not self.children:
            print(self.key)
            return
        self.children.sort(key=lambda x: x.key)

        if self.key < self.children[0].key:
            print(self.key)
            for i in self.children:
                i.order()
        else:
            for i in sorted(self.children + [Node(self.key)], key=lambda x: x.key):
                if i.key == self.key:
                    print(self.key)
                else:
                    i.order()

d = {}
t = int(input())
for i in range(t):
    s = list(map(int, input().split()))
    if s[0] not in d.keys():
        d[s[0]] = Node(s[0])
        head[s[0]] = True
    root = d[s[0]]
    for j in s[1:]:
        if j not in d.keys():
            d[j] = Node(j)
        head[j] = False
        root.children.append(d[j])
root = None
for key, val in head.items():
    if val:
        root = d[key]
        break
root.order()

```



![image-20240404123708386](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240404123708386.png)





## 2. 学习总结和收获

本次机考较简单，树的题目递归建树已经得心应手了，队列、栈、堆等题目除了一些模板题目外，使用python内置库是非常好的选择。另外这些题目和字典等结构的结合是很普遍的。less or equal提醒我们要注意极端情况和边际值。最后一题读题出现问题导致浪费时间，下次应该注意，不能太着急把题读完，细心读题，否则会错失关键信息。

建议之后命题时加强数据，卡掉不正确的时间复杂度的做法，或者像数算a机考一样采用IOI赛制，使得暴力只能拿到20~30%的分数，这样才能抛弃计概的根据题意暴力模拟的思维。





