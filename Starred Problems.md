# Monotonous Stack

## Notes

A monotonic stack is a stack whose elements are monotonically increasing or decreasing. It maintains monotonicity while popping elements when a new item is pushed into the stack.

The monotonic stack problem is mainly **the previous/next smaller/larger problem**.

## Sample Problem (OJ 22067: 快速堆猪)

### Solutions

#### Monotonous Stack

In this case, this monotonous stack in also called auxiliary stack (辅助栈).

```python
stack = []
auxiliary_stack = []  # a descending monotonous stack

while True:
    try:
        command = input().split()
        if command[0] == "pop":
            if stack:
                stack.pop()
                if auxiliary_stack:
                    auxiliary_stack.pop()
        elif command[0] == "min":
            if auxiliary_stack:
                print(auxiliary_stack[-1])
        else:
            pushing_value = int(command[1])
            stack.append(pushing_value)
            if not auxiliary_stack:
                auxiliary_stack.append(pushing_value)
            else:
                # due to the feature of stack, we only need to 
                # store to minimum value.
                auxiliary_stack.append(min(auxiliary_stack[-1], pushing_value))
    except EOFError:
        break

```

#### Lazy Deletion

Using min heap to maintain the minimum value, and do not update it unless minimum value is called.

```python
import heapq


class PigStack:
    def __init__(self):
        self.stack = []
        self.min_heap = []  # min heap
        self.popped = []

    def push(self, weight):
        self.stack.append(weight)
        heapq.heappush(self.min_heap, weight)

    def pop(self):
        if self.stack:
            weight = self.stack.pop()
            self.popped.append(weight)

    def min(self):
        # update min_heap only when called
        while self.min_heap and self.min_heap[0] in self.popped:
            heapq.heappop(self.min_heap)  # lazy deletion
            self.popped.pop()
        if self.min_heap:
            return self.min_heap[0]
        else:
            return None


pig_stack = PigStack()
while True:
    try:
        command = input().split()
        if command[0] == 'push':
            pig_stack.push(int(command[1]))
        elif command[0] == 'pop':
            pig_stack.pop()
        elif command[0] == 'min':
            min_weight = pig_stack.min()
            if min_weight is not None:
                print(min_weight)
    except EOFError:
        break

```

## Related Problems

### [LeetCode 496. Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)

Maintain a monotonous stack, but need to pop smaller values in it.

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        hashmap = {}
        mono_stack = []
        for i in nums2[::-1]:
            while mono_stack and i > mono_stack[-1]:
                mono_stack.pop()
            if not mono_stack:
                hashmap[i] = -1
            else:
                hashmap[i] = mono_stack[-1]
            mono_stack.append(i)
        return [hashmap[i] for i in nums1]
    
```

# Merge Sort

## Sample Problem (OJ 02299: Ultra-QuickSort)

```python
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

# Prefix, Infix and Postfix Notation

## Notes

### Evaluating

#### Prefix
1. Read the prefix expression from right to left.
2. Start with an empty stack. When you encounter an operand, push it onto the stack.
3. When you encounter an operator, pop the required number of operands from the stack (depending on the operator, typically two for binary operators), apply the operator to those operands, and push the result back onto the stack.
4. At the end, the stack should contain one element, which is the result of the expression.
#### Infix
Infix notation is more complex to evaluate directly, as you need to account for operator precedence and parentheses. This usually involves converting the infix expression to postfix expression first and then evaluating it using the postfix evaluation method.
#### Postfix
1. Read the postfix expression from left to right.
2. Start with an empty stack. When you encounter an operand, push it onto the stack.
3. When you encounter an operator, pop the required number of operands from the stack, apply the operator to those operands in the correct order, and push the result back onto the stack.
4. At the end, the stack should contain one element, which is the result of the expression.

### Converting

#### Prefix

##### Prefix to Infix

1. Read the prefix expression from right to left.
2. Start with an empty stack. When you see an operand, push it onto the stack.
3. When you see an operator, pop the top two items from the stack, which will be operands, and combine them into a single string with the operator between them, and parentheses around them. Then push this string back onto the stack.
4. At the end, the stack will contain a single element, which is the equivalent infix expression.

##### Prefix to Postfix

1. Read the prefix expression from right to left.
2. Start with an empty stack. When you see an operand, push it onto the stack.
3. When you see an operator, pop the top two items from the stack, which will be operands. Combine them into a single string with the operands first, followed by the operator, without parentheses. Then push this string back onto the stack.
4. At the end, the stack will contain a single element, which is the equivalent postfix expression.

#### Infix

##### Infix to Prefix

1. Reverse the infix expression.
2. Replace every '(' with ')' and every ')' with '('.
3. Convert the resulting expression to postfix notation.
4. Reverse the postfix expression to get the prefix notation.

##### Infix to Postfix

1. Start with an empty stack. Iterate over the infix expression.
2. When you see an operand, add it to the output.
3. When you see an operator, pop from the stack all operators that have higher or equal precedence and add them to the output, then push the current operator onto the stack.
4. If you see '(', push it onto the stack. If you see ')', pop operators from the stack and add them to the output until you pop '('.
5. After the end of the expression, pop and output all remaining operators from the stack.

#### Postfix

##### Postfix

1. Read the postfix expression from left to right.
2. Start with an empty stack. When you see an operand, push it onto the stack.
3. When you see an operator, pop the top two items from the stack, which will be operands. Combine them into a single string with the operator in front and then push this string back onto the stack.
4. At the end, the stack will contain a single element, which is the equivalent prefix expression.

##### Postfix

1. Read the postfix expression from left to right.
2. Start with an empty stack. When you see an operand, push it onto the stack.
3. When you see an operator, pop the top two items from the stack, which will be operands. Wrap them with parentheses and place the operator between them. Then push this string back onto the stack.
4. At the end, the stack will contain a single element, which is the equivalent infix expression.

## Sample Problems

