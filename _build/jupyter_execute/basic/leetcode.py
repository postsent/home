#!/usr/bin/env python
# coding: utf-8

# # Leetcode
# 
# Below is the optimal solution, my attempt is in GitHub - another file in the same directory.

# # Potential References  
# https://github.com/CyC2018/CS-Notes

# # --Basic--

# # --Algorithm--

# # Binary Search

# ## search

# In[25]:


"""
assume ascending order, unique
"""
# https://leetcode.com/problems/binary-search/solution/
from typing import * # import typing, for python3.9, see: https://stackoverflow.com/questions/57505071/nameerror-name-list-is-not-defined

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            pivot = left + (right - left) // 2 # the middle between left & right then add the offset of current left position
            if nums[pivot] == target:
                return pivot
            if target < nums[pivot]:
                right = pivot - 1
            else:
                left = pivot + 1
        return -1
s = Solution()
res = s.search(nums=[1,2,3], target=1)
res


# ## firstBadVersion

# In[ ]:


# https://leetcode.com/problems/first-bad-version/submissions/

# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 1, n 
        first_bad = 0
        if n == 1:
            return 1
        while left <= right:
            pivot = left + (right - left) // 2
            
            if isBadVersion(pivot):
                first_bad = pivot # store current, if next is not bad, use this
                right = pivot - 1
            else:
                left = pivot + 1
                
        return first_bad


# ## searchInsert

# In[ ]:


class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left,right = 0, len(nums) -1
        is_left = True 
        # whether should insert at same index if target smaller than last compare element after finish binary search, 
        # otherwise insert at the last compared element index + 1
        p = 0
        
        if not nums: # empty
            return 0
        
        while left <= right:
            p = left + (right - left) // 2
            if target > nums[p]:
                left = p + 1
                is_left = False
            elif target < nums[p]:
                right = p - 1
                is_left = True
            else:
                is_left = True
                break
        if is_left:
            return p
        return p + 1


# # Two pointers

# ## sortedSquares
# 
# [link](https://leetcode.com/problems/squares-of-a-sorted-array/discuss/1642628/Python-O(n)-Solution-or-Two-Pointers)  
# Two pointers. One pointer is at index 0 and the other at last index.     
# We execute while loop till first index is smaller than the last index. At every loop we check if the sqaure of number at first index (nums[i] ** 2) is greater than number at last index (nums[j] ** 2).   
# If square of number at index i (first pointer) is greater than square of number at index j (second pointer). We add nums[i] ** 2 to the result array, otherwise we add nums[j] ** 2 to the array.  
# Lastly, reverse the array.  
# O(2n) = O(n).

# In[ ]:


def sortedSquares(self, nums: List[int]) -> List[int]:       
        
        i,j = 0, len(nums)-1
        result = []
        while i < j:
            a,b = nums[i] ** 2 , nums[j] ** 2
            if a > b:
                result.insert(0,a)
                i+=1
            else:
                result.insert(0,b)
                j-=1
        result.insert(0, nums[i] ** 2)
        
        return result


# In[7]:


x


# ## Rotate array

# In[ ]:


class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        k =  k % len(nums)  # if rotate exceed the len, then remove that full rotation and keep the remaining steps
        split = len(nums)-k            
        left = nums[0: split]
        right = nums[split:]
        nums[0:k] = right
        nums[k:] = left


# ## Move Zeroes
# https://leetcode.com/problems/move-zeroes/  
# Input: nums = [0,1,0,3,12]  
# Output: [1,3,12,0,0]

# One pointer to the zero, one iterate.  
# When meet non-zero, then swap inex with  with the zero.  
# The order of non-zeros are maintained since the swap order is same as the existing order.

# In[8]:


class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        zero_pointer = 0
        for i, el in enumerate(nums):
            print("zero_pointer, i, el, nums:", zero_pointer, i, el, nums)
            if el != 0:
                nums[zero_pointer], nums[i] = nums[i], nums[zero_pointer]
                zero_pointer += 1
                
s = Solution()
nums=[0,1,2,4,3]
s.moveZeroes(nums)
nums


# ## twoSums

# Since the pointers go from outside in, when reach one of the correct one, the pointer will
# not move any more inner as the values do not matched (sorted in non-descent).  
# 
# Example: target = 6;  
# 
# .... 2 ..... 4 .....  
# 
# **Case 1:** pointer1 reach 2, pointer2 after 4.   
# Now, the value is always greater, and so pointer1 will not move.  
# **Case 2:** pointer1 before 2, pointer2 reach 4.  
# Now, the value is always smaller, so pointer2 will not move. And pointer1 will move until reach 2.

# In[ ]:


# ref: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/discuss/1642733/Python-Two-Simple-Approaches-or-Binary-Search-and-Two-Pointers
# Two Pointers Solution
# Time O(n)
class Solution:
	def twoSum(self, arr: List[int], target: int) -> List[int]:
		strt = 0
		end = len(arr)-1
		while strt <= end:
			sum = arr[strt]+arr[end]
			if sum == target:
				return [strt+1, end+1]
			elif sum < target:
				strt += 1
			else:
				end -= 1


# ## reverseString

# In[ ]:


## Reverse String
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s) - 1
        
        while left <= right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1 


# or s.reseve() # not good

# # Linked List

#  ## Middle of the Linked List

# In[ ]:


# ref: https://leetcode.com/problems/middle-of-the-linked-list/solution/
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        arr = [head]
        while arr[-1].next:
            arr.append(arr[-1].next)
        return arr[len(arr) // 2]


# In[ ]:


class Solution:
    def middleNode(self, head):
        """
        When traversing the list with a pointer slow, make another pointer fast that traverses twice as fast. When fast reaches the end of the list, slow must be in the middle.
        """
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow


# ## removeNthFromEnd

# In[ ]:


def removeNthFromEnd(head, n):
    
    def remove(head, n):            
        # base case
        if head == None: 
            return head, 0 
        # recursive til the end
        node, count = remove(head.next, n)
        # stack pop back up, and start counting 
        count += 1
        # update head
        head.next = node
        
        if count == n: # since count starts from 0, so use the next one instead once reach n
            head = head.next
        
        return head, count
    
    return remove(head, n)[0]
nums = [1,2,3,4,5]
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# head = ListNode()
# p = head
# i = 0
# while i < len(nums):
#     p.val = nums[i]
#     if i + 1 == len(nums):
#         a.next = None
#         break
#     p.next = nums[i+1]
#     p = p.next
#     i += 1 
    
# removeNthFromEnd(a)


# # Other

# ## isPalindrome

# In[ ]:


# with str
x = -121

x = str(x)
x == x[::-1]
                 


# In[ ]:


# without str: https://leetcode.com/problems/palindrome-number/solution/
# sidenotes, in C, if declare type int, then the rest are all int but python need to cast explicitly
x = 54321 # 12345

if x < 0 or (x % 10 == 0 and x != 0): # negative or last digit == 0 except for 0
    print(False)
    
reverted = 0
while x > reverted:
    reverted = reverted * 10 + x % 10 # everytime the last digit will increment by 1 position
    x /= 10
    x = int(x)
    reverted = int(reverted)
    print("x:", x)
    print("reverted:", reverted)

# // When the length is an odd number, we can get rid of the middle digit by revertedNumber/10
# // For example when the input is 12321, at the end of the while loop we get x = 12, revertedNumber = 123,
# // since the middle digit doesn't matter in palidrome(it will always equal to itself), we can simply get rid of it.
x == reverted or x == int(reverted/10) 


# # TODO
