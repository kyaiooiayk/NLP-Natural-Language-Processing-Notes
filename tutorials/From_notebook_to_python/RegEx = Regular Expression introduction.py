#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#What-are-regular-expression?" data-toc-modified-id="What-are-regular-expression?-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>What are regular expression?</a></span></li><li><span><a href="#Regex-functions" data-toc-modified-id="Regex-functions-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Regex functions</a></span></li><li><span><a href="#Metacharacters" data-toc-modified-id="Metacharacters-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Metacharacters</a></span></li><li><span><a href="#Special-sequences" data-toc-modified-id="Special-sequences-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Special sequences</a></span></li><li><span><a href="#Sets" data-toc-modified-id="Sets-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Sets</a></span></li><li><span><a href="#References" data-toc-modified-id="References-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# **What?** RegEx = Regular Expression introduction
# 
# </font>
# </div>

# # Imports

# In[2]:


import re


# # What are regular expression?

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - The methods of Python's ``str`` type give you a powerful set of tools for formatting, splitting, and manipulating string data. 
# - But a more powerful tool is -> built-in **regular expression** module.
# - Regular expressions are a means of *flexible pattern matching* in strings.
# - If you frequently use the command-line, you are probably familiar with this type of flexible matching with the "``*``" character, which acts as a wildcard.
# 
# </font>
# </div>

# # Regex functions

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - **findall** 	Returns a list containing all matches
# - **search** 	Returns a Match object if there is a match anywhere in the string
# - **split** 	Returns a list where the string has been split at each match
# - **sub** 	Replaces one or many matches with a string
# 
# </font>
# </div>

# In[2]:


# Look into the available methods
print(dir(re))


# # Metacharacters

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Metacharacters are characters with a special meaning 
# 
# </font>
# </div>

# ![image.png](attachment:image.png)

# **Find all lower case characters alphabetically between "a" and "m":**

# In[4]:


txt = "The rain in Spain"

x = re.findall("[a-m]", txt)
print(x)


# **Find all digit characters:**

# In[5]:


txt = "That will be 59 dollars"
x = re.findall("\d", txt)
print(x)


# **Search for a sequence that starts with "he", followed by two (any) characters, and an "o":**

# In[3]:


txt = "hello world"
x = re.findall("he..o", txt)
print(x)


# **Return anything that starts with "he", followed by any characters, and an "o":**

# In[24]:


txt = "hello world"
x = re.findall("he*.*l", txt)
print(x)


# **Check if the string starts with 'hello':**

# In[7]:


txt = "hello world"
x = re.findall("^hello", txt)
if x:
    print("Yes, the string starts with 'hello'")
else:
    print("No match")


# **Check if the string ends with 'world':**

# In[8]:


txt = "hello world"

x = re.findall("world$", txt)
if x:
    print("Yes, the string ends with 'world'")
else:
    print("No match")


# **Check if the string contains "ai" followed by 0 or more "x" characters:**

# In[ ]:


txt = "The rain in Spain falls mainly in the plain!"

x = re.findall("aix*", txt)
print(x)
if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string contains "ai" followed by 1 or more "x" characters:**

# In[10]:


txt = "The rain in Spain falls mainly in the plain!"
x = re.findall("aix+", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string contains "a" followed by exactly two "l" characters:**

# In[11]:


txt = "The rain in Spain falls mainly in the plain!"
x = re.findall("al{2}", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string contains either "falls" or "stays":**

# In[12]:


txt = "The rain in Spain falls mainly in the plain!"
x = re.findall("falls|stays", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# # Special sequences

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - A special sequence is a \ followed by one of the characters in the list below, and has a special meaning: 
# 
# </font>
# </div>

# ![image.png](attachment:image.png)

# **Check if the string starts with "The":**

# In[13]:


txt = "The rain in Spain"
x = re.findall("\AThe", txt)
print(x)

if x:
    print("Yes, there is a match!")
else:
    print("No match")


# **Check if "ain" is present at the beginning of a WORD:**

# In[15]:


txt = "The rain in Spain"
x = re.findall(r"\bain", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if "ain" is present at the end of a WORD:**

# In[16]:


txt = "The rain in Spain"
x = re.findall(r"ain\b", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if "ain" is present, but NOT at the beginning of a word:**

# In[17]:


txt = "The rain in Spain"
x = re.findall(r"\Bain", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if "ain" is present, but NOT at the end of a word:**

# In[18]:


txt = "The rain in Spain"
x = re.findall(r"ain\B", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string contains any digits (numbers from 0-9):**

# In[19]:


txt = "The rain in Spain"
x = re.findall("\d", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Return a match at every no-digit character:**

# In[20]:


txt = "The rain in Spain"
x = re.findall("\D", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Return a match at every white-space character:**

# In[21]:


txt = "The rain in Spain"
x = re.findall("\s", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Return a match at every NON white-space character:**

# In[22]:


txt = "The rain in Spain"
x = re.findall("\S", txt)

print(x)
if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Return a match at every word character (characters from a to Z, digits from 0-9, and the underscore _ character):**

# In[23]:


txt = "The rain in Spain"
x = re.findall("\w", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Return a match at every NON word character (characters NOT between a and Z. Like "!", "?" white-space etc.):**

# In[25]:


import re
txt = "The rain in Spain"
x = re.findall("\W", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string ends with "Spain":**

# In[26]:


txt = "The rain in Spain"
x = re.findall("Spain\Z", txt)
print(x)

if x:
    print("Yes, there is a match!")
else:
    print("No match")


# # Sets

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - A set is a set of characters inside a pair of square brackets [] with a special meaning:
# 
# </font>
# </div>

# ![image.png](attachment:image.png)

# **Check if the string has any a, r, or n characters:**

# In[27]:


txt = "The rain in Spain"
x = re.findall("[arn]", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string has any characters between a and n:**

# In[28]:


txt = "The rain in Spain"
x = re.findall("[a-n]", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string has other characters than a, r, or n:**

# In[ ]:


txt = "The rain in Spain"
x = re.findall("[^arn]", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string has any 0, 1, 2, or 3 digits:**

# In[29]:


txt = "The rain in Spain"
x = re.findall("[0123]", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string has any digits:**

# In[30]:


txt = "8 times before 11:45 AM"
x = re.findall("[0-9]", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string has any two-digit numbers, from 00 to 59:**

# In[31]:


txt = "8 times before 11:45 AM"
x = re.findall("[0-5][0-9]", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string has any characters from a to z lower case, and A to Z upper case:**

# In[32]:


txt = "8 times before 11:45 AM"
x = re.findall("[a-zA-Z]", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# **Check if the string has any + characters:**

# In[33]:


txt = "8 times before 11:45 AM"
x = re.findall("[+]", txt)
print(x)

if x:
    print("Yes, there is at least one match!")
else:
    print("No match")


# # References

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# - https://www.w3schools.com/python/python_regex.asp
# 
# </font>
# </div>

# In[ ]:




