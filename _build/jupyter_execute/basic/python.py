#!/usr/bin/env python
# coding: utf-8

# # Python

# # Potential References  
# https://github.com/CyC2018/CS-Notes

# # --Basic--

# # --Algorithm--

# # Binary Search

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


class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        # all positive -> same order
        # all negative -> reverse order
        # mix -> two pointers
        
        # check postive
        pos = 0
        neg = 0
        
        for n in nums:
            if n >= 0:
                pos += 1
            else:
                neg += 1
        
        if not nums: # empty
            return nums
        
        if len(nums) == 1:
            return [n**2 for n in nums]
        
        if pos == 0: # all neg
            reversed = nums[::-1]
            squared = [n**2 for n in reversed]
            return squared
        elif neg == 0:
            squared = [n**2 for n in nums]
            return squared
        
        
        first_pos = 0
        
        # find first pos
        for i, n in enumerate(nums):
            if n >= 0:
                first_pos = i
                break
        
        p1, p2 = first_pos - 1, first_pos
        
        sorted = []
        
        # two pointers, one post one neg
        while p1 >= 0 and p2 < len(nums):
            v_neg = nums[p1]
            v_pos = nums[p2]
            if v_pos <= abs(v_neg):
                sorted.append(v_pos)
                p2 += 1 
            else:
                sorted.append(v_neg)
                p1 -= 1 
                
        if p1 >= 0: # some neg left
            sorted += nums[:p1+1][::-1] # add remain and reverse since negative values
            return [n**2 for n in sorted]
        
        elif p2 < len(nums): # some pos left
            sorted += nums[p2:]
            return [n**2 for n in sorted]
        
        # nothing left
        return [n**2 for n in sorted]
            
        # worse case O(3n) = O(n)
s = Solution()
res = s.sortedSquares(nums=[0,2])
res


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
