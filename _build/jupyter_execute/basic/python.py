#!/usr/bin/env python
# coding: utf-8

# # Python

# # Basic

# # Algorithm

# ## isPalindrome

# In[42]:


# with str
x = -121

x = str(x)
x == x[::-1]
                 


# In[43]:


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


# ## Binary Search

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


# In[3]:





# # TODO
