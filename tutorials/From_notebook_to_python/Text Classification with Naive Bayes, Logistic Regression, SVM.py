#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Text Classification with Naive Bayes, Logistic Regression, SVM

Reference: https://github.com/practical-nlp/practical-nlp/blob/master/Ch4/01_OnePipeline_ManyClassifiers.ipynb
"""


# # Import modules

# In[ ]:


import numpy as np
import pandas as pd #to work with csv files
import matplotlib as mpl 
import matplotlib.cm as cm 
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
import string
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn import metrics
from time import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Getting rid of the warning messages
import warnings
warnings.filterwarnings("ignore")


# # Loda dataset

# In[ ]:


"""
We will be using a dataset called "Economic news article tone and relevance" from Figure-Eight which consists of 
approximately 8000 news articles, which were tagged as relevant or not relevant to the US Economy.
"""


# In[ ]:


our_data = pd.read_csv("./Full-Economic-News-DFE-839861.csv" , encoding = "ISO-8859-1" )

display(our_data.shape) #Number of rows (instances) and columns in the dataset
our_data["relevance"].value_counts()/our_data.shape[0] #Class distribution in the dataset


# In[ ]:


"""
There is an imbalance in the data with **not relevant** being 82% in the dataset. That is, most of the articles 
are not relevant to US Economy, which makes sense in a real-world scenario, as news articles discuss various topics. 
We should keep this class imbalance mind when interpreting the classifier performance later. Let us first convert the
class labels into binary outcome variables for convenience. 1 for Yes (relevant), and 0 for No (not relevant), and 
ignore "Not sure". 
"""


# In[ ]:


# convert label to a numerical variable
our_data = our_data[our_data.relevance != "not sure"]
our_data.shape
our_data['relevance'] = our_data.relevance.map({'yes':1, 'no':0}) #relevant is 1, not-relevant is 0. 
our_data = our_data[["text","relevance"]] #Let us take only the two columns we need.
our_data.shape


# # Text Pre-processing

# In[ ]:


"""
Typical steps involve tokenization, lower casing, removing, stop words, punctuation markers etc, and vectorization. 
Other processes such as stemming/lemmatization can also be performed. Here, we are performing the following steps: 
removing br tags, punctuation, numbers, and stopwords. While we are using sklearn's list of stopwords, there are 
several other stop word lists (e.g., from NLTK) or sometimes, custom stopword lists are needed depending on the task. 
"""


# In[ ]:


stopwords = stop_words.ENGLISH_STOP_WORDS
def clean(doc): #doc is a string of text
    doc = doc.replace("</br>", " ") #This text contains a lot of <br/> tags.
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = " ".join([token for token in doc.split() if token not in stopwords])
    #remove punctuation and numbers
    return doc


# # Train-test split

# In[ ]:


"""
Now we are ready for the modelling. We are going to use algorithms from sklearn package. We will go through the 
following steps:

1 Split the data into training and test sets (75% train, 25% test)    
2 Extract features from the training data using CountVectorizer, which is a bag of words feature  implementation. 
  We will use the pre-processing function above in conjunction with Count Vectorizer  
3 Transform the test data into the same feature vector as the training data.  
4 Train the classifier  
5 Evaluate the classifier 
"""


# In[ ]:


#Step 1: train-test split
X = our_data.text #the column text contains textual data to extract features from
y = our_data.relevance #this is the column we are learning to predict. 
print(X.shape, y.shape)
# split X and y into training and testing sets. By default, it splits 75% training and 25% test
#random_state=1 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


#Step 2-3: Preprocess and Vectorize train and test data
vect = CountVectorizer(preprocessor=clean) #instantiate a vectoriezer
X_train_dtm = vect.fit_transform(X_train)#use it to extract features from training data
#transform testing data (using training data's features)
X_test_dtm = vect.transform(X_test)
print(X_train_dtm.shape, X_test_dtm.shape)
#i.e., the dimension of our feature vector is 49753!


# # Helper function

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)


# # Multinomial Naive Bayes model

# In[ ]:


#Step 3: Train the classifier and predict for test data
nb = MultinomialNB() #instantiate a Multinomial Naive Bayes model
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)#train the model(timing it with an IPython "magic command")')
y_pred_class = nb.predict(X_test_dtm)#make class predictions for X_test_dtm


# In[ ]:


#Step 4: Evaluate the classifier using various measures

# Function to plot confusion matrix. 
# Ref:http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
from sklearn.metrics import roc_auc_score
  
    
#Print accuracy:
print("Accuracy: ", accuracy_score(y_test, y_pred_class))

    
# print the confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Not Relevant','Relevant'],normalize=True,
                      title='Confusion matrix with all features')

# calculate AUC: Area under the curve(AUC) gives idea about the model efficiency:
#Further information: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
print("ROC_AOC_Score: ", roc_auc_score(y_test, y_pred_prob))


# In[ ]:


"""
At this point, we can notice that the classifier is doing poorly with identifying relevant articles, while it is
doing well with non-relevant ones. Our large feature vector could be creating a lot of noise in the form of very
rarely occurring features that are not useful for learning. Let us change the count vectorizer to take a certain
number of features as maximum. 
"""


# In[ ]:


vect = CountVectorizer(preprocessor=clean, max_features=5000) #Step-1
X_train_dtm = vect.fit_transform(X_train)#combined step 2 and 3
X_test_dtm = vect.transform(X_test)
nb = MultinomialNB() 
# Train the model(timing it with an IPython "magic command")
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')
# Make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred_class))
# Print the confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Not Relevant','Relevant'],normalize=True,
                      title='Confusion matrix with max 5000 features')


# In[ ]:


"""
Clearly, the performance on relevance classification got better even though the overall accuracy fell by 10%. 
"""


# # Logistic regression

# In[ ]:


"""
Let us try another classification algorithm and see if the performance changes. For this experiment, we have 
considered logistic regression, with class_weight attribute as "balanced", to address the problem of class 
imbalance in this dataset. 
"""


# In[ ]:


logreg = LogisticRegression(class_weight="balanced") #instantiate a logistic regression model
logreg.fit(X_train_dtm, y_train) #fit the model with training data

#Make predictions on test data
y_pred_class = logreg.predict(X_test_dtm)

#calculate evaluation measures:
print("Accuracy: ", accuracy_score(y_test, y_pred_class))
print("AUC: ", roc_auc_score(y_test, y_pred_prob))
cnf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Not Relevant','Relevant'],normalize=True,
                      title='Confusion matrix with normalization')


# # SVC - Support Vector Classifier

# In[ ]:


"""
Let us wrap this notebook by trying with one more classifier, but reducing the feature vector size to 1000.
"""


# In[ ]:


vect = CountVectorizer(preprocessor=clean, max_features=1000) #Step-1
X_train_dtm = vect.fit_transform(X_train)#combined step 2 and 3
X_test_dtm = vect.transform(X_test)

classifier = LinearSVC(class_weight='balanced') #instantiate a logistic regression model
classifier.fit(X_train_dtm, y_train) #fit the model with training data

#Make predictions on test data
y_pred_class = classifier.predict(X_test_dtm)

#calculate evaluation measures:
print("Accuracy: ", accuracy_score(y_test, y_pred_class))
print("AUC: ", roc_auc_score(y_test, y_pred_prob))
cnf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Not Relevant','Relevant'],normalize=True,
                      title='Confusion matrix with normalization')


# # Conclusion

# In[ ]:


"""
So, how do we choose whats the best? 

 - If we look at overall accuracy alone, we should be choosing the very first classifier in this notebook. However, 
   that is also doing poorly with identifying "relevant" articles. 
 - If we choose purely based on how good it is doing with "relevant" category, we should choose the second one we
   built. 

So, what to choose as the best among these depends on what we are looking for in our usecase! 
"""

