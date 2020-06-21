#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm , metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import model_selection


# In[2]:


#estrazione dati dal dataset 
users = pd.read_excel('keystroke_51_Aggiornato.xls', 'users')
timers = pd.read_excel('keystroke_51_Aggiornato.xls', 'timers')
utenti = timers.join(users.set_index('UserName'), on='UserName')


# In[3]:


utenti.shape


# In[4]:


#Selezione delle colonne di interesse 
df_utenti = pd.DataFrame(utenti,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age'])
df_utenti.shape


# In[5]:


df_utenti.columns


# In[6]:


df_utenti.info()


# In[7]:


#suddivide il dataset per età 
età=df_utenti['Age'].value_counts()
print(età)
explode = (0, 0,0.1) 
età.plot.pie(explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90, counterclock=False)


# In[8]:


#Utilizzo della libreria seaborn per la creazione del boxplot tra SommaPP e Age
sns.boxplot(x='Age', y='MediaPP', data=df_utenti)


# In[9]:


#Utilizzo della libreria seaborn per la creazione del boxplot tra SommaPR e Age
sns.boxplot(x='Age', y='MediaPR',data=df_utenti)


# In[10]:


#Utilizzo della libreria seaborn per la creazione del boxplot tra SommaRR e Age
sns.boxplot(x='Age', y='MediaRR',data=df_utenti)


# In[11]:


#Utilizzo della libreria seaborn per la creazione del boxplot tra SommaRP e Age
sns.boxplot(x='Age', y='MediaRP',data=df_utenti)


# In[12]:


sns.pairplot(df_utenti, hue = 'Age', palette = 'Dark2')


# In[13]:


#metodo per la rimozione dei valori nan
df_utenti = df_utenti.fillna(method='ffill')


# In[14]:


#divisione dataset per la funzione train_test_split
df_test = pd.DataFrame(df_utenti,columns = ['MediaPP','MediaPR','MediaRR','MediaRP'])
y = df_utenti['Age']


# In[15]:


df_test.info()


# In[16]:


#preprocessing dei dati 
scaler = StandardScaler()
df_test = scaler.fit_transform(df_test)


# In[17]:


#divisione del dataset in 70 train e 30 test
X_train, X_test, y_train, y_test = train_test_split(df_test, y, test_size=0.3, random_state = 42, shuffle = True)


# In[18]:


print(df_test.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[20]:


classifier = RandomForestClassifier(max_depth=5, n_estimators=100,random_state=42,criterion='entropy')


# In[21]:


classifier.fit(X_train, y_train)


# In[22]:


predicted = classifier.predict(X_test)


# In[23]:


predicted.shape
y_test.shape


# In[24]:


print(y_test.values)


# In[25]:


print(X_test)


# In[26]:


print(predicted)


# In[27]:


## Accuracy score 
print(accuracy_score(y_test, predicted))


# In[28]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[19]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per RandomForestClassifier
parameters2 = {'n_estimators': [10, 20, 50, 100],
              'bootstrap': [True, False],
              'criterion': ["gini", "entropy"],
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters2)
clf.fit(df_test,y)
clf.best_estimator_


# In[29]:


#Evaluate a score by cross-validation
cv_results = model_selection.cross_val_score(clf,df_test,y) 
print (cv_results.mean())


# In[ ]:




