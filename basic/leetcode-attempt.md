# My attempts

# Two pointers
## sortedSquares
```py3
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
```
## twoSum
```py
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        for idx, el in enumerate(numbers):
            remain = target - el
            # binary search on this value
            res = self.binarySearch(numbers, remain, idx)
            if res != -1:
                return [idx+1, res+1]
            
    def binarySearch(self, nums, target, cur):
        left, right = 0, len(nums) - 1
        while left <= right:
            
            pivot = left + (right - left) // 2 
            
            if nums[pivot] == target:
                if cur == pivot:
                    # special case: same value e.g. 4 + 4 = 8
                    idx = pivot
                    while idx >= 0:
                        if nums[idx] != target:
                            idx += 1
                            break
                        idx -= 1
                        
                    while idx < len(nums):
                        if idx != cur:
                            return idx
                        idx += 1
                return pivot
            if target < nums[pivot]:
                right = pivot - 1
            else:
                left = pivot + 1
        return -1
s = Solution()
s.twoSum([1,2,3,4,4,9,56,90], 8)
```