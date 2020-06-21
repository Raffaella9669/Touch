#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import svm , metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn import model_selection
from numpy import mean



# In[2]:


#estrazione dati dal dataset 
users = pd.read_excel('keystroke_51_Aggiornato.xls', 'users')
timers = pd.read_excel('keystroke_51_Aggiornato.xls', 'timers')
utenti = timers.join(users.set_index('UserName'), on='UserName')


# In[3]:


#Selezione delle colonne di interesse 
df_utenti = pd.DataFrame(utenti,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age'])
df_utenti.shape


# In[4]:


df_Age1 = df_utenti[df_utenti['Age'] == 1]
df_Age2 = df_utenti[df_utenti['Age'] == 2] 
df_Age3 = df_utenti[df_utenti['Age'] == 3]
#print(df_Age)


# In[5]:


#Divisione dataset per bilanciare la categoria 2 
df21 = df_Age2.iloc[:209]
df22 = df_Age2.iloc[209:418]
df23 = df_Age2.iloc[321:530]


# In[6]:


#Lavoriamo sul primo dataset creato
Age_split = np.concatenate([df_Age1,df21,df_Age3])
Age_split.shape


# In[7]:


df_age_split = pd.DataFrame(Age_split,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split'])
print(df_age_split)


# In[8]:


df_age_split.info()


# In[9]:


#suddivide il dataset per età 
età=df_age_split['Age_split'].value_counts()
print(età)
explode = (0, 0,0.1) 
età.plot.pie(explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90, counterclock=False)


# In[10]:


#creazione del boxplot tra MediaPP e Age 
df_age_split.boxplot(column='MediaPP', by='Age_split',figsize=(8,8))


# In[11]:


#creazione del boxplot tra MediaPR e Age 
df_age_split.boxplot(column='MediaPR', by='Age_split',figsize=(8,8))


# In[12]:


#creazione del boxplot tra MediaRR e Age 
df_age_split.boxplot(column='MediaRR', by='Age_split',figsize=(8,8))


# In[13]:


#creazione del boxplot tra MediaRP e Age 
df_age_split.boxplot(column='MediaRP', by='Age_split',figsize=(8,8))


# In[14]:


df_test = df_age_split[['MediaPP','MediaPR','MediaRR','MediaRP']]
y = df_age_split['Age_split']


# In[ ]:


y


# In[ ]:


df_test.info()


# In[ ]:


sns.pairplot(df_age_split, hue = 'Age_split', palette = 'Dark2')


# In[ ]:


#Accuracy 0.64
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_test = scaler.fit_transform(df_test)


# In[15]:


#Accuracy 0.64 Preprocessing dati
scaler = preprocessing.StandardScaler().fit(df_test)
df_test = scaler.transform(df_test)


# In[ ]:


#Accuracy 0.61
min_max_scaler = preprocessing.MinMaxScaler()
df_test = min_max_scaler.fit_transform(df_test)


# In[ ]:


#Accuracy 0.64
from sklearn.preprocessing import MaxAbsScaler
transformer = MaxAbsScaler().fit(df_test)
df_test = transformer.transform(df_test)


# In[ ]:


#Accuracy 0.42
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
df_test = normalizer.fit_transform(df_test)


# In[ ]:


#Accuracy 0.60
from sklearn.preprocessing import normalize
df_test = normalize(df_test)
#df_test = normalizer.fit_transform(df_test)


# In[ ]:


#Accuracy 0.63
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(df_test)
df_test = transformer.transform(df_test)
df_test


# In[16]:


import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle= True, random_state=42)
kf.get_n_splits(df_test)

print(kf)
for train_index, test_index in kf.split(df_test):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = df_test[train_index], df_test[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[ ]:


print(df_test.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[21]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per svm.SVC
parameters = {'C':[0.1,1,10,100,1000],
              'gamma':('auto','scale'),
              'kernel': ('linear','rbf','sigmoid'),
              'decision_function_shape':('ovo','ovr')
             }
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=kf)
clf.fit(df_test,y)
clf.best_estimator_


# In[57]:


accuracy1 = clf.best_score_
accuracy1


# In[23]:


#Evaluate a score by cross-validation
cv_results=model_selection.cross_val_score(clf,df_test,y,cv=kf) 
print (cv_results.mean())


# In[24]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per RandomForestClassifier
parameters2 = {'n_estimators': [10, 20, 50, 100],
              'bootstrap': [True, False],
              'criterion': ["gini", "entropy"],
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf = RandomForestClassifier()
clf1 = GridSearchCV(rf, parameters2, cv=kf)
clf1.fit(df_test,y)
clf1.best_estimator_


# In[27]:


accuracy2 = clf1.best_score_
accuracy2


# In[28]:


#Evaluate a score by cross-validation
cv_results2 = model_selection.cross_val_score(clf1,df_test,y, cv=kf) 
print (cv_results2.mean())


# In[29]:


#Lavoriamo sul secondo dataset creato
Age_split2 = np.concatenate([df_Age1,df22,df_Age3])
Age_split2.shape


# In[30]:


df_age_split2 = pd.DataFrame(Age_split2,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split'])
print(df_age_split2)


# In[31]:


df_test2 = df_age_split2[['MediaPP','MediaPR','MediaRR','MediaRP']]
y2 = df_age_split2['Age_split']


# In[32]:


#Preprocessing dati
scaler = preprocessing.StandardScaler().fit(df_test2)
df_test2 = scaler.transform(df_test2)


# In[33]:


kf2 = KFold(n_splits=5, shuffle= True, random_state=42)
kf2.get_n_splits(df_test2)

print(kf2)
for train_index, test_index in kf2.split(df_test2):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train2, X_test2 = df_test2[train_index], df_test2[test_index]
    y_train2, y_test2 = y2[train_index], y2[test_index]


# In[34]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per svm.SVC
parameters3 =  {'C':[0.1,1,10,100,1000],
              'gamma':('auto','scale'),
              'kernel': ('linear','rbf','sigmoid'),
              'decision_function_shape':('ovo','ovr')
             }
svc2 = svm.SVC()
clf3 = GridSearchCV(svc2, parameters3, cv=kf2)
clf3.fit(df_test2,y2)
clf3.best_estimator_


# In[36]:


accuracy3 = clf3.best_score_
accuracy3


# In[37]:


#Evaluate a score by cross-validation
cv_results3 = model_selection.cross_val_score(clf3,df_test2,y2, cv=kf2) 
print (cv_results3.mean())


# In[38]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per RandomForestClassifier
parameters4 = {'n_estimators': [10, 20, 50, 100],
              'bootstrap': [True, False],
              'criterion': ["gini", "entropy"],
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf2 = RandomForestClassifier()
clf4 = GridSearchCV(rf2, parameters4, cv=kf2)
clf4.fit(df_test2,y2)
clf4.best_estimator_


# In[40]:


accuracy4 = clf4.best_score_
accuracy4


# In[41]:


#Evaluate a score by cross-validation
cv_results4 = model_selection.cross_val_score(clf4,df_test2,y2,cv=kf2) 
print (cv_results4.mean())


# In[42]:


#Lavoriamo sul terzo dataset creato
Age_split3 = np.concatenate([df_Age1,df23,df_Age3])
Age_split3.shape


# In[43]:


df_age_split3 = pd.DataFrame(Age_split3,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split'])
print(df_age_split3)


# In[44]:


df_test3 = df_age_split3[['MediaPP','MediaPR','MediaRR','MediaRP']]
y3 = df_age_split3['Age_split']


# In[45]:


scaler = preprocessing.StandardScaler().fit(df_test3)
df_test3 = scaler.transform(df_test3)


# In[46]:


kf3 = KFold(n_splits=5, shuffle= True, random_state=42)
kf3.get_n_splits(df_test3)

print(kf3)
for train_index, test_index in kf3.split(df_test3):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train3, X_test3 = df_test3[train_index], df_test3[test_index]
    y_train3, y_test3 = y3[train_index], y3[test_index]


# In[47]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per svm.SVC
parameters5 = {'C':[0.1,1,10,100,1000],
              'gamma':('auto','scale'),
              'kernel': ('linear','rbf','sigmoid'),
              'decision_function_shape':('ovo','ovr')
             }
svc3 = svm.SVC()
clf5 = GridSearchCV(svc3, parameters5, cv = kf3)
clf5.fit(df_test3,y3)
clf5.best_estimator_


# In[48]:


accuracy5 = clf5.best_score_
accuracy5


# In[49]:


#Evaluate a score by cross-validation
cv_results5 = model_selection.cross_val_score(clf5,df_test3,y3,cv=kf3) 
print (cv_results5.mean())


# In[50]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per RandomForestClassifier
parameters6 = {'n_estimators': [10, 20, 50, 100],
              'bootstrap': [True, False],
              'criterion': ["gini", "entropy"],
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf3 = RandomForestClassifier()
clf6 = GridSearchCV(rf3, parameters6,cv=kf3)
clf6.fit(df_test3,y3)
clf6.best_estimator_


# In[55]:


accuracy6 = clf6.best_score_
accuracy6


# In[52]:


#Evaluate a score by cross-validation
cv_results6 = model_selection.cross_val_score(clf6,df_test3,y3,cv=kf3) 
print (cv_results6.mean())


# In[58]:


#Calcolo media dell'accuracy dei 3 dataset per la SVM
accuracySVM = [accuracy1,accuracy3,accuracy5]
np.mean(accuracySVM)


# In[56]:


#Calcolo media dell'accuracy dei 3 dataset per RF
accuracyRF= [accuracy2,accuracy4,accuracy6]
np.mean(accuracyRF)


# In[ ]:




