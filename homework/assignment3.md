# Assignment #3: March月考

Updated 1622 GMT+8 Mar 9, 2024

2024 spring, Complied by 张朕源 元培学院 2200017763

**编程环境**

操作系统：Windows 11 22621.3155

Python编程环境：PyCharm 2022.1.4 (Community Edition)



## 1. 题目

**02945: 拦截导弹**

http://cs101.openjudge.cn/practice/02945/



思路：最长非增子序列，动归



##### 代码

```python
# 
n = int(input())
l = list(map(int, input().split()))
dp = [0] * n
for i in range(n - 1, -1, -1):
    max_n = 1
    for j in range(n - 1, i, -1):
        if l[i] >= l[j]:
            max_n = max(dp[j] + 1, max_n)
    dp[i] = max_n
print(max(dp))
```

![image-20240309162258771](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240309162258771.png)





**04147:汉诺塔问题(Tower of Hanoi)**

http://cs101.openjudge.cn/practice/04147



思路：递归



##### 代码

```python
# 
n, a, b, c = map(str, input().split())
n = int(n)
def hanoi(num, home, temp, target, name=None):
    if num == 1:
        print(f'{name}:{home}->{target}')
        return
    hanoi(num-1, home, target, temp, num-1)
    hanoi(1, home, temp, target, num)
    hanoi(num-1, temp, home, target, num-1)
hanoi(n, a, b, c)

```



![image-20240309162337958](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240309162337958.png)





**03253: 约瑟夫问题No.2**

http://cs101.openjudge.cn/practice/03253



思路：模拟



##### 代码

```python
# 
def joseph(n, p, m):
    lst = list(range(p, n + 1)) + list(range(1, p))
    num, ans = 0, ''
    while lst:
        num = (num + m - 1) % len(lst)
        ans += ',' + str(lst.pop(num))
    return ans.lstrip(',')


while True:
    n, p, m = map(int, input().split())
    if n == 0 and p == 0 and m == 0:
        break
    print(joseph(n, p, m))
```



![image-20240309162510589](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240309162510589.png)





**21554:排队做实验 (greedy)v0.2**

http://cs101.openjudge.cn/practice/21554



思路：用排序实现的贪心



##### 代码

```python
# 
n = int(input())
lst = sorted(enumerate(list(map(int, input().split()))), key=lambda x: x[-1])
ans = 0
for i in range(len(lst)):
    ans += lst[i][-1] * (n - 1 - i)
    print(lst[i][0] + 1, end=' ')
print('\n%.2f' % (ans / n))
```



![image-20240309162413015](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240309162413015.png)





**19963:买学区房**

http://cs101.openjudge.cn/practice/19963



思路：排序



##### 代码

```python
# 
def median(lst, k):
    m = len(lst)
    lst.sort(key=k)
    return k(lst[m // 2]) if m % 2 != 0 else (k(lst[m // 2 - 1]) + k(lst[m // 2])) / 2


n = int(input())
distances = [[sum(map(int, i.split(',')))] for i in [i[1:-1] for i in input().split()]]
prices = list(map(int, input().split()))
for i in range(n):
    distances[i] = [prices[i], distances[i][0] / prices[i]]
price_median, cp_median = median(distances, lambda x: x[0]), median(distances, lambda x: x[-1])
ans = 0
for i in distances:
    if i[0] < price_median and i[1] > cp_median:
        ans += 1
print(ans)
```



![image-20240309162527126](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240309162527126.png)





**27300: 模型整理**

http://cs101.openjudge.cn/practice/27300



思路：排序



##### 代码

```python
# 
llm = {}
for i in range(int(input())):
    name, num = map(str, input().split('-'))
    if name in llm:
        llm[name].append(num)
    else:
        llm[name] = [num]
for name in sorted(list(llm.keys())):
    print(name, end=': ')
    print(', '.join(sorted(llm[name], key=lambda x: float(x[:-1]) if x[-1] == 'M' else float(x[:-1]) * 1000)))
```



![image-20240309162610440](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20240309162610440.png)





## 2. 学习总结和收获

刷每日选做剩下的题，学习AVL树和拓扑排序，在做数算pre的内容





