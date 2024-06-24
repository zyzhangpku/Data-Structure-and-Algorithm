# Assignment #2: 编程练习

Updated 1600 GMT+8 Mar 1, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763



**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：

简单的类的应用，魔法方法

##### 代码

```python
# 
class fraction():
    def __init__(self, num, den):
        self.num = int(str(num))
        self.den = int(str(den))

    def __str__(self):
        return f"{self.num}/{self.den}"

    def __add__(self, otherFraction):
        newtop = self.num * otherFraction.den + self.den * otherFraction.num
        newden = self.den * otherFraction.den
        common = gcd(newtop, newden)
        return fraction(newtop // common, newden // common)

    
def gcd(m, n):
    while m % n != 0:
        oldm, oldn = m, n
        m, n = oldn, oldm % oldn
    return n


s = list(map(int, input().split()))
a, b = fraction(s[0], s[1]), fraction(s[2], s[3])
print(a + b)
```



![image-20240301165146962](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240301165146962.png)





### 04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110



思路：

以为是背包做了半天不对，结果是贪心，破防了

##### 代码

```python
# 
qq = list(map(int, input().split()))
n, w = qq[0], qq[1]
sb = []
for i in range(n):
    qqq = list(map(int, input().split()))
    sb.append(qqq)
sb.sort(reverse=True, key=lambda x: x[0] / x[1])
gib = 0
for i in sb:
    if i[1] <= w:
        gib += i[0]
        w -= i[1]
    else:
        gib += i[0] / i[1] * w
        w = 0
print('%.1f'%gib)
```



![image-20240301164320481](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240301164320481.png)





### 18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/



思路：

字典+排序

##### 代码

```python
# 
nn = int(input())


def da():
    l = list(map(int, input().split()))
    n, m, xueliang = l[0], l[1], l[2]
    h = {}
    tl = []
    for i in range(n):
        q = list(map(int, input().split()))
        if q[0] in h.keys():
            h[q[0]].append(q[1])
        else:
            h[q[0]] = [q[1]]
            tl.append(q[0])
    for i in h.values():
        i.sort(reverse=True)
    tl.sort()
    for x in tl:
        zz = 0
        while zz < m and zz < len(h[x]):
            xueliang -= h[x][zz]
            zz += 1
            if xueliang <= 0:
                return x
    return 'alive'


for _ in range(nn):
    print(da())
```



![image-20240301164343905](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240301164343905.png)







### 18176: 2050年成绩计算

http://cs101.openjudge.cn/practice/18176/



思路：

筛素数

##### 代码

```python
# 
from math import sqrt
N = 10005

s = [True] * N
p = 2
while p * p <= N:
	if s[p]:
		for i in range(p * 2, N, p):
			s[i] = False
	p += 1

m, n = [int(i) for i in input().split()]

for i in range(m):
	x = [int(i) for i in input().split()]
	sum = 0
	for num in x:
		root = int(sqrt(num))
		if num > 3 and s[root] and num == root * root:
			sum += num
	sum //= len(x)
	if sum == 0:
		print(0)
	else:
		print('%.2f' % sum)
```



![image-20240301194511347](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240301194511347.png)





## 2. 学习总结和收获

做了一些每日选做中的二叉树的简单题目。



