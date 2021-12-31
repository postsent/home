#!/usr/bin/env python
# coding: utf-8

# # Python

# # isPalindrome

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


# In[ ]:


109 // 10


# # TODO
