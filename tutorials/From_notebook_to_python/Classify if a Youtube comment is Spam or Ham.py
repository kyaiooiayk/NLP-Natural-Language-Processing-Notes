#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Fetch-and-load-the-dataset" data-toc-modified-id="Fetch-and-load-the-dataset-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Fetch and load the dataset</a></span></li><li><span><a href="#Data-wrangling" data-toc-modified-id="Data-wrangling-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data wrangling</a></span></li><li><span><a href="#Regex-Based-Labeling-Functions" data-toc-modified-id="Regex-Based-Labeling-Functions-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Regex Based Labeling Functions</a></span></li><li><span><a href="#Writing-more-labeling-functions" data-toc-modified-id="Writing-more-labeling-functions-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Writing more labeling functions</a></span></li><li><span><a href="#Outputs" data-toc-modified-id="Outputs-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Outputs</a></span></li><li><span><a href="#Training-a-classifier" data-toc-modified-id="Training-a-classifier-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Training a classifier</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# In[ ]:


"""
What? Classify if a Youtube comment is Spam or Ham

"""


# In[ ]:


"""
!python -m spacy download en_core_web_sm
"""


# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[2]:


import warnings
warnings.filterwarnings('ignore')
import os
import wget
import zipfile
import shutil
import re
import glob
import utils
from snorkel.analysis import get_label_buckets
from snorkel.labeling import labeling_function
from snorkel.labeling import LFAnalysis
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LabelingFunction
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling.lf.nlp import nlp_labeling_function
from snorkel.preprocess import preprocessor
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.utils import probs_to_preds
from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Fetch and load the dataset
# <hr style = "border:2px solid black" ></hr>

# In[ ]:


"""
Let's get the Youtube spam classification dataset from the UCI ML Repository archive.
"""


# In[6]:


file_link = "http://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"

datasetPath = "YouTube-Spam-Collection_dataset"
os.makedirs(datasetPath, exist_ok=True)
if not os.path.exists(os.path.join(datasetPath, "YouTube-Spam-Collection-v1.zip")):
    wget.download(file_link, out=datasetPath)
else:
    print("File already exists")

with zipfile.ZipFile(os.path.join(datasetPath, "YouTube-Spam-Collection-v1.zip"), 'r') as zip_ref:
    zip_ref.extractall(datasetPath)

shutil.rmtree(os.path.join(datasetPath, "__MACOSX"))
os.remove(os.path.join(datasetPath, "YouTube-Spam-Collection-v1.zip"))
os.listdir(datasetPath)


# # Data wrangling
# <hr style = "border:2px solid black" ></hr>

# In[7]:


def load_spam_dataset(load_train_labels: bool = False, split_dev_valid: bool = False):
    filenames = sorted(glob.glob(os.path.join(datasetPath, "Youtube*.csv")))
    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        # Lowercase column names
        df.columns = map(str.lower, df.columns)
        # Remove comment_id field
        df = df.drop("comment_id", axis=1)
        # Add field indicating source video
        df["video"] = [i] * len(df)
        # Rename fields
        df = df.rename(columns={"class": "label", "content": "text"})
        # Shuffle order
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    df_train = pd.concat(dfs[:4])
    df_dev = df_train.sample(100, random_state=123)

    if not load_train_labels:
        df_train["label"] = np.ones(len(df_train["label"])) * -1
    df_valid_test = dfs[4]
    df_valid, df_test = train_test_split(
        df_valid_test, test_size=250, random_state=123, stratify=df_valid_test.label
    )

    if split_dev_valid:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_test


# In[11]:


# Let's look at the train and test dataset
df_train, df_test = load_spam_dataset()


# In[13]:


print("Train")
display(df_train.head(5))


# In[14]:


print('Test')
df_test.head(5)


# In[17]:


Y_train = df_train.label.values
Y_train[:5]


# In[22]:


# the train lavel seems to have only abstain value?
df_train.label.unique()


# In[21]:


df_test.label.unique()


# In[16]:


Y_test = df_test.label.values
Y_test[:5]


# In[ ]:


"""
There are a few things to keep in mind with respect to the dataset:
    HAM represents a NON-SPAM comment.
    SPAM is a SPAM comment
    ABSTAIN is for neither of the above

We initialise their respective values below:
    ABSTAIN = -1
    HAM = 0
    SPAM = 1
"""


# In[21]:


ABSTAIN = -1
HAM = 0
SPAM = 1


# In[ ]:


"""
We need to find some pattern in the data, so as to create rules for labeling the data.
Hence, we randomly display some rows of the dataset so that we can try to find some pattern in the text.
"""


# In[30]:


df_train[["author", "text", "video", "label"]].sample(20, random_state=2020)


# In[24]:


"""
An example for how a labeling function can be defined , anything that might match the pattern of a spam message
can be used, here http is an example to show how spam comments may contain links
"""


# In[18]:


@labeling_function()
def lf_contains_link(x):
    # Return a label of SPAM if "http" in comment text, otherwise ABSTAIN
    return SPAM if "http" in x.text.lower() else ABSTAIN


# In[ ]:


"""
Defining labeling functions to check strings such as 'check', 'check out', 'http', 'my channel', 'subscribe'
"""


# In[19]:


@labeling_function()
def check(x):
    return SPAM if "check" in x.text.lower() else ABSTAIN


@labeling_function()
def check_out(x):
    return SPAM if "check out" in x.text.lower() else ABSTAIN


@labeling_function()
def my_channel(x):
    return SPAM if "my channel" in x.text.lower() else ABSTAIN


@labeling_function()
def if_subscribe(x):
    return SPAM if "subscribe" in x.text.lower() else ABSTAIN


# In[ ]:


"""
Using LFApplier to use our labelling functions with pandas dataframe object.
These labeling functions can also be used on columns other than 'text'.
"""


# In[22]:


lfs = [check_out, check, lf_contains_link, my_channel, if_subscribe]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)


# In[24]:


L_train


# In[ ]:


"""
Coverage is the fraction of the dataset the labeling function labels.
"""


# In[25]:


coverage_check_out, coverage_check, coverage_link, coverage_my_channel, coverage_subscribe= (L_train != ABSTAIN).mean(axis=0)
print(f"check_out coverage: {coverage_check_out * 100:.1f}%")
print(f"check coverage: {coverage_check * 100:.1f}%")
print(f"link coverage: {coverage_link * 100:.1f}%")
print(f"my_channel coverage: {coverage_my_channel * 100:.1f}%")
print(f"if_subscribe coverage: {coverage_subscribe * 100:.1f}%")


# In[ ]:


"""
Before we procees further, let us understand a bit of jargon with respect to the summary of the LFAnalysis

    Polarity - set of unique labels that the labeling function outputs (excluding Abstains)
    Overlaps - where there is at least one common entry for more than one labeling functions i.e the labeling
               fucntions agree upon the value to be returned
    Conflicts - where the labeling functions disagree upon the value to be returned
"""


# In[26]:


LFAnalysis(L=L_train, lfs=lfs).lf_summary()


# In[ ]:


"""
Trying and checking the results by filtering out the matching rows and checking for false positives
"""


# In[27]:


# display(df_train.iloc[L_train[:, 1] == SPAM].sample(10, random_state=2020))
# display(df_train.iloc[L_train[:, 2] == SPAM].sample(10, random_state=2020))
df_train.iloc[L_train[:, 3] == SPAM].sample(10, random_state=2020)


# In[ ]:


"""
Combining two labeling functions and checking the results
"""


# In[28]:


#buckets = get_label_buckets(L_train[:, 0], L_train[:, 1])
#buckets = get_label_buckets(L_train[:, 1], L_train[:, 2])
buckets = get_label_buckets(L_train[:, 0], L_train[:, 3])

df_train.iloc[buckets[(ABSTAIN, SPAM)]].sample(10, random_state=1)


# # Regex Based Labeling Functions
# <hr style = "border:2px solid black" ></hr>

# In[ ]:


"""
Using regular expressions to make the labeling functions more adaptive over 
differnt variations of the pattern string and repeating the same process as above.
"""


# In[29]:


#using regular expressions
@labeling_function()
def regex_check_out(x):
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN


# In[30]:


lfs = [check_out, check, regex_check_out]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)


# In[31]:


LFAnalysis(L=L_train, lfs=lfs).lf_summary()


# In[33]:


buckets = get_label_buckets(L_train[:, 1], L_train[:, 2])
df_train.iloc[buckets[(SPAM, ABSTAIN)]].sample(10, random_state=2020)


# In[ ]:


"""
Let's use a 3rd party model, TextBlob in this case, to write a labeling function. Snorkel makes this very
simple to implement.
"""


# In[34]:


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x


# In[35]:


@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return HAM if x.polarity > 0.9 else ABSTAIN


# In[36]:


@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    return HAM if x.subjectivity >= 0.5 else ABSTAIN


# In[37]:


lfs = [textblob_polarity, textblob_subjectivity]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)


# In[38]:


LFAnalysis(L_train, lfs).lf_summary()


# # Writing more labeling functions
# <hr style = "border:2px solid black" ></hr>

# In[ ]:


"""
Single labeling functions arent enough to test the entire databsase with accuracy as they do not have enough
coverage, we usually need to combine differnt labeling functions(more rubost and accurate ones) to get this done.

Keyword based labeling fucntions: These are similar to the ones used befeore with the labeling_fucntion decorator.
here we just make a few changes.
"""


# In[39]:


def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords, label=SPAM):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )


"""Spam comments talk about 'my channel', 'my video', etc."""
keyword_my = make_keyword_lf(keywords=["my"])

"""Spam comments ask users to subscribe to their channels."""
keyword_subscribe = make_keyword_lf(keywords=["subscribe"])

"""Spam comments post links to other channels."""
keyword_link = make_keyword_lf(keywords=["http"])

"""Spam comments make requests rather than commenting."""
keyword_please = make_keyword_lf(keywords=["please", "plz"])

"""Ham comments actually talk about the video's content."""
keyword_song = make_keyword_lf(keywords=["song"], label=HAM)


# In[ ]:


"""
Modifying the above functions to use regualr expressions too would be an interesting exercise which we will leave 
to the reader.

Having other methods such as a Rule of Thumb or Heuristics(length of text) could help too. These are not extremely 
accurate but will get the job done to a certain extent.

An example is given below
"""


# In[40]:


@labeling_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.text.split()) < 5 else ABSTAIN


# In[ ]:


"""
We can also use NLP preprocessors such as spaCy to enrich our data and provide us with more fields to work on 
which will make the labeling a bit easier 
"""


# In[41]:


# The SpacyPreprocessor parses the text in text_field and
# stores the new enriched representation in doc_field
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

@labeling_function(pre=[spacy])
def has_person(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN


# In[42]:


#snorkel has a pre built labeling function like decorator that uses spaCy as it is a very common nlp preprocessor

@nlp_labeling_function()
def has_person_nlp(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN


# # Outputs
# <hr style = "border:2px solid black" ></hr>

# In[ ]:


"""
Let's move onto learning how we can go about combining labeling function outputs with labeling models.
"""


# In[43]:


lfs = [
    keyword_my,
    keyword_subscribe,
    keyword_link,
    keyword_please,
    keyword_song,
    regex_check_out,
    short_comment,
    has_person_nlp,
    textblob_polarity,
    textblob_subjectivity,
]


# In[44]:


applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)


# In[45]:


LFAnalysis(L=L_train, lfs=lfs).lf_summary()


# In[ ]:


"""
We plot a histogram to get an idea about the coverages of the labeling functions
"""


# In[46]:


def plot_label_frequency(L):
    plt.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]))
    plt.xlabel("Number of labels")
    plt.ylabel("Fraction of dataset")
    plt.show()

plot_label_frequency(L_train)


# In[ ]:


"""
We now convert the labels from our labeling functions to a single noise-aware probabilistic label per data. 
We do so by taking a majority vote on what the data should be labeled as .i.e if more labeling functions agree 
that the text/data is spam , then we label it as spam
"""


# In[47]:


majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)


# In[48]:


preds_train


# In[ ]:


"""
However there may be functions that are correlated and might give a false sense of majority , to handle this we
use a differernt snorkel label model to comine inputs of the labeling functions.
"""


# In[49]:


label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)


# In[50]:


majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")


# In[ ]:


"""
We plot another graph to see the confidences that each data point is a spam
"""


# In[51]:


def plot_probabilities_histogram(Y):
    plt.hist(Y, bins=10)
    plt.xlabel("Probability of SPAM")
    plt.ylabel("Number of data points")
    plt.show()

probs_train = label_model.predict_proba(L=L_train)
plot_probabilities_histogram(probs_train[:, SPAM])


# In[ ]:


"""
There might be some data which do not get any label from the functions , we filter them out as follows
"""


# In[53]:


df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train)


# # Training a classifier
# <hr style = "border:2px solid black" ></hr>

# In[ ]:


"""
In this section we use the probabilistic training labels we generated to train a classifier. 
For demonstration we use Scikit-Learn.
"""


# In[54]:


vectorizer = CountVectorizer(ngram_range=(1, 5))
X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())


# In[55]:


preds_train_filtered = probs_to_preds(probs=probs_train_filtered)


# In[56]:


sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
sklearn_model.fit(X=X_train, y=preds_train_filtered)


# In[57]:


print(f"Test Accuracy: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")


# # References
# <hr style = "border:2px solid black" ></hr>

# - Reference: https://github.com/practical-nlp/practical-nlp/blob/master/Ch2/06_Snorkel.ipynb

# In[ ]:




