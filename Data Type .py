#!/usr/bin/env python
# coding: utf-8

# # Data Types

# ## 1. mutable
# 
#    *List
#    *Set
#    *Dictionary
#     
# ## 2. immutable
# 
#    *Number
#    *String
#    *Tuple
#    *frozon set

# # LIST
List is a mutable data type
A collection of values stored within a square bracket
Duplicate values are allowed in list
We can all types of data stored in list [].Example:
list = [1,2.3,4+5j,'words',1,2.3]
1 is int.
2.3 is float.
4+5j is complex.
'words' is string.
1,2.3 is duplicate value.
# ## checking data type

# In[1]:


List = [1,2.3,4+5j,'name',1]
print(List) # PRINT IS A OUTPUT FUNCTION.LIST IS VARIABLE.
print(type(List)) # The type() function is used to determine the type of an object.it return the type of variable.


# # How access the list element using index position?
Two type of indexing are available.
1.Forword indexing.
2.Backword indexing.
# ![Screenshot%202024-03-10%20074453.png](attachment:Screenshot%202024-03-10%20074453.png)

# # Inbuild function of list in python

# In[2]:


list1 = [10,20,30,40,50,60,60,78]
print(type(list1))
print('list is,',list1)


# # append(): Adds an element to end of the list

# In[3]:


list1.append(34)
print('result: add 34 to end of this list')
print(list1)


# ## Extend():adds elements from another list to the end of current list.

# In[4]:


list2 =[98,23,44]
list1.extend(list2)
print(list1)


# # Insert():inserts an element at spacified position.

# In[5]:


list1.insert(1,5)
list1


# # Remove(): remove the first occurence of a spacified element in the list

# In[6]:


list1.remove(23)
list1


# # Pop: removes and returns the element at the spacified position.

# In[7]:


list1.pop()
list1


# ## Index(): return the index of the first occurence spacified element 

# In[8]:


print(list1.index(50))


# # Count():returns the number of occurence of the spacified element.

# In[9]:


print(list1.count(60))


# # Sort(): sorts the list in accending order (by default) 

# In[10]:


list1.sort()
list1


# # Reverse(): reverse the element of the list in place

# # list1.reverse()
# list1

# # Copy(): returns shallow copy of the list.

# In[11]:


new_list = list1.copy()
print(new_list)


# # Aggregate function in list

# # sum(): calculate the sum of all  element in list.

# In[12]:


print('sum of the list is',sum(list1))


# ## Max(): find maximum value in the list.

# In[13]:


print('maximum value of the list is:',max(list1))


# ## Min(): find minimum value in the list.

# In[14]:


print('minimum of value the list is:',min(list1))


# # Len(): returns the number of element in the list.

# In[15]:


print('lenth of the list is:', len(list1))


# ## Any and All: check if any or all in the list evaluate to true.

# In[16]:


print(any(list1))
print(all(list1))


# # Statistics functions in list:

# mean(): calculates arithmetic mean of the list.

# In[17]:


from statistics import mean
print(mean(list1))


# median(): calculate the median (meddle value) of the list.

# In[18]:


from statistics import median
print(median(list1))


# mode(): calculates the mode of the list (most common value)

# In[19]:


from statistics import mode
print(mode(list1))


# varience(): calculates variance of the list

# In[20]:


from statistics import variance
print(variance(list1))


# stdev(): calculates standard diviation of the list

# In[21]:


from statistics import stdev
print(stdev(list1))


# In[22]:


list1


# ## slicing - [start values: end value: step values]
# 

# In[32]:


print(list1[0:])
print(list1[:-4])
print(list1[1:5])
print(list1[:7])
print(list1[::-1])
print(list1[::4])
print(list1[3:8:2])
print(list1[:-1])


# # Delete: delete element in the list:

# In[36]:


del list1[4]
print(list1)


# ## clear():clear all element in a list

# In[37]:


list1.clear()
list1

