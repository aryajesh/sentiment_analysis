#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import nltk
import pickle
import re
import string
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix

from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from collections import defaultdict
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


# # Accessing the cleaned Dataset

# In[3]:


df=pd.read_csv('/Users/aryajeshkumar/Desktop/project2020/data.csv')
df.shape


# In[4]:


import seaborn as sns

sns.countplot(x='sentiment', data=df)


# In[5]:


df.head()
df["review"][3]


# In[6]:


X = []
sentences = list(df['review'])
for sen in sentences:
    X.append((sen))
y = df['sentiment']
X[3]


# # Vectorization

# In[7]:


vectorizer = TfidfVectorizer(min_df=30,
                             strip_accents = None,
                             lowercase = False,
                             preprocessor = None,
                             use_idf = True,
                             norm = 'l2',
                             smooth_idf = True)
bow = vectorizer.fit_transform(df['review'])
labels = df['sentiment']
#print(bow)


# In[8]:


with open('vectorizer.pkl', 'wb') as f:
    joblib.dump(vectorizer, f)


# In[9]:


len(vectorizer.get_feature_names())


# # Splitting the Dataset: The Train and Test Sets

# In[10]:


x_train,x_test,y_train,y_test=train_test_split(bow,labels,
                                               random_state=1,
                                              test_size=0.33,
                                              shuffle = True)
x_train.shape


# In[11]:


x_test.shape


# In[ ]:





# # Logistic Regression Classifier

# In[12]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(x_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)


# In[27]:


from sklearn.externals import joblib

clf=LogisticRegressionCV(solver='lbfgs',cv=5,
                        scoring = 'accuracy',
                        random_state=None,
                        n_jobs=-1,
                        verbose=3).fit(x_train,y_train)
with open('sentiment_classifier.pkl', 'wb') as f:
    joblib.dump(clf, f)
model_columns = list(x_train.columns)
joblib.dump(model_columns,'model_columns.pkl')


# # Accuracy 

# In[14]:


clf.score(x_test,y_test)


# In[15]:


y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred)


# In[16]:


import matplotlib.pyplot as plt
cm = confusion_matrix(y_test,y_pred) 

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


# # SVM Classifier & Accuracy

# In[17]:


svc = LinearSVC()
svc.fit(x_train, y_train)
svc.score(x_test,y_test)


# In[18]:


y_pred = svc.predict(x_test)
confusion_matrix(y_test, y_pred)
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test,y_pred) 

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


# # Making Prediction for a single instance 

# In[19]:


root = '/Users/aryajeshkumar/Desktop/project2020/test'
filename = 'index.txt'
file = open(root+ '/' +filename, 'r')
text = file.read()
print(text)


# In[20]:


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence.lower())
    
    # stopwords removal
    sentence = stop_words(sentence)
    
    # lemmatization 
    sentence = lemmatize_words(sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def stop_words(text):
    filtered_sentence = []
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        if w not in stopwords.words('english') and w.isalpha():
            filtered_sentence.append(w)
            str1=' '.join(str(e) for e in filtered_sentence)
    return str1

def lemmatize_words(text):
    tokens = nltk.word_tokenize(text)
    lmtzr = WordNetLemmatizer()
    tagged=nltk.pos_tag(tokens)
    filtered_sentence = []
    for token, tag in tagged:
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        filtered_sentence.append(lemma)
        str2=' '.join(str(e) for e in filtered_sentence)
    return str2 


# In[21]:


instance = preprocess_text(text)
print(instance)


# In[22]:


threshold = 0.5
clf.fit(bow,labels)

trial_review = vectorizer.transform([instance])

sentence = vectorizer.transform(['this is a bad movie'])

prob = clf.predict_proba(sentence)
#prob = clf.predict_proba(trial_review)

#prob = clf.predict(trial_review)
print(prob)

if prob[0][0] >= threshold:
    print("Negative Review")
else:
    print("Positive Review")


# In[ ]:





# In[ ]:




