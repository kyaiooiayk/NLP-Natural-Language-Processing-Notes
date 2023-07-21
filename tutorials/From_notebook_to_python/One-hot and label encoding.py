#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? One-hot and label encoding

We'll code through two examples:
    [1] Manual implementation
    [2] Scikit-learn implementation
    
Reference: https://github.com/practical-nlp/practical-nlp/blob/master/Ch3/01_OneHotEncoding.ipynb
"""


# # Import libraries/modules

# In[15]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# # Manual implementation of One Hot Encoding

# In[1]:


documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]
processed_docs = [doc.lower().replace(".","") for doc in documents]
processed_docs


# In[2]:


#Build the vocabulary
vocab = {}
count = 0
for doc in processed_docs:
    for word in doc.split():
        if word not in vocab:
            count = count +1
            vocab[word] = count
print(vocab)


# In[ ]:


"""
Get one hot representation for any string based on this vocabulary. 
If the word exists in the vocabulary, its representation is returned. 
If not, a list of zeroes is returned for that word. 
"""


# In[5]:


def get_onehot_vector(somestring):
    onehot_encoded = []
    for word in somestring.split():
        temp = [0]*len(vocab)
        if word in vocab:
            # -1 is to take care of the fact indexing in array starts from 0 and not 1
            temp[vocab[word]-1] = 1 
        onehot_encoded.append(temp)
    return onehot_encoded


# In[ ]:


"""
We expect the dimension of the matrix to be [3x6] = [No words in sentece x No words in vocabulary]
"""


# In[12]:


print(processed_docs[1])
#one hot representation for a text from our corpus.
get_onehot_vector(processed_docs[1]) 


# In[13]:


get_onehot_vector("man and dog are good") 
#one hot representation for a random text, using the above vocabulary


# In[14]:


get_onehot_vector("man and man are good") 


# # One hot vs. label encoding using scikit-learn

# In[ ]:


"""
One Hot Encoding: In one-hot encoding, each word w in corpus vocabulary is given a unique integer id wid that is
                  between 1 and |V|, where V is the set of corpus vocab. Each word is then represented by a 
                  V-dimensional binary vector of 0s and 1s.

Label Encoding: In Label Encoding, each word w in our corpus is converted into a numeric value between 0 and n-1
                (where n refers to number of unique words in our corpus).
"""


# In[16]:


S1 = 'dog bites man'
S2 = 'man bites dog'
S3 = 'dog eats meat'
S4 = 'man eats food'


# In[36]:


data = [S1.split(), S2.split(), S3.split(), S4.split()]
values = data[0]+data[1]+data[2]+data[3]
print(data)
print("The data: ",values)
print("The unique data: ",set(values))


# In[37]:


#One-Hot Encoding
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(data).toarray()
print("Onehot Encoded Matrix:\n",onehot_encoded)


# In[38]:


#Label Encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print("Label Encoded:",integer_encoded)

