#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm , metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn import model_selection
from numpy import mean
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#estrazione dati dal dataset 
users = pd.read_excel('keystroke_51_Aggiornato.xls', 'users')
timers = pd.read_excel('keystroke_51_Aggiornato.xls', 'timers')
utenti = timers.join(users.set_index('UserName'), on='UserName')


# In[3]:


#Selezione delle colonne di interesse 
df_utenti = pd.DataFrame(utenti,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age','Gender','trialsattempted'])
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


df_age_split = pd.DataFrame(Age_split,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split','Gender','trialsattempted'])
print(df_age_split)


# In[8]:


df_age_split


# In[9]:


df_age_split.info()


# In[10]:


#suddivide il dataset per età 
età=df_age_split['Age_split'].value_counts()
print(età)
explode = (0, 0,0.1) 
età.plot.pie(explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90, counterclock=False)


# In[11]:


#creazione del boxplot tra MediaPP e Age 
df_age_split.boxplot(column='MediaPP', by='Age_split',figsize=(8,8))


# In[12]:


#creazione del boxplot tra MediaPR e Age 
df_age_split.boxplot(column='MediaPR', by='Age_split',figsize=(8,8))


# In[13]:


#creazione del boxplot tra MediaRR e Age 
df_age_split.boxplot(column='MediaRR', by='Age_split',figsize=(8,8))


# In[14]:


#creazione del boxplot tra MediaRP e Age 
df_age_split.boxplot(column='MediaRP', by='Age_split',figsize=(8,8))


# In[15]:


df_test = df_age_split[['MediaPP','MediaPR','MediaRR','MediaRP','Gender','trialsattempted']]
y = df_age_split['Age_split']


# In[16]:


y


# In[17]:


df_test.info()


# In[18]:


#Preprocessing dei dati 
scaler = preprocessing.StandardScaler().fit(df_test)
df_test = scaler.transform(df_test)


# In[19]:


#Divisione train(70) e test (30)
X_train, X_test, y_train, y_test = train_test_split(df_test, y, test_size=0.3, shuffle = True, random_state = 42)


# In[20]:


df_test


# In[21]:


print(df_test.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[22]:


#utilizzo del classificazione svm.SVC
classifier = svm.SVC(C=10,decision_function_shape='ovo',gamma='scale')


# In[23]:


classifier.fit(X_train, y_train)


# In[24]:


predicted = classifier.predict(X_test)


# In[25]:


predicted.shape
y_test.shape


# In[26]:


print(predicted)


# In[27]:


#Accuracy score 
accuracy1 = accuracy_score(y_test, predicted)
print(accuracy1)


# In[28]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[29]:


#utilizzo di RandomForestClassifier 
classifier2 = RandomForestClassifier(max_depth=10,max_features=4, random_state=42,n_estimators=20) 


# In[30]:


classifier2.fit(X_train, y_train)


# In[31]:


predicted2 = classifier2.predict(X_test)


# In[32]:


#Accuracy score 
accuracy2 = accuracy_score(y_test, predicted2)
print(accuracy2)


# In[33]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier2, metrics.classification_report(y_test, predicted2)))
disp = metrics.plot_confusion_matrix(classifier2, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[34]:


#Lavoriamo sul secondo dataset creato
Age_split2 = np.concatenate([df_Age1,df22,df_Age3])
Age_split2.shape


# In[35]:


df_age_split2 = pd.DataFrame(Age_split2,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split','Gender','trialsattempted'])
print(df_age_split2)


# In[36]:


df_test2 = df_age_split2[['MediaPP','MediaPR','MediaRR','MediaRP','Gender','trialsattempted']]
y2 = df_age_split2['Age_split']


# In[37]:


scaler = preprocessing.StandardScaler().fit(df_test2)
df_test2 = scaler.transform(df_test2)


# In[38]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(df_test2, y2, test_size=0.3, random_state=0, shuffle = True )


# In[39]:


classifier3 = svm.SVC(C=10,decision_function_shape='ovo',gamma='auto')#a support vector classifier


# In[40]:


classifier3.fit(X_train2, y_train2)


# In[41]:


predicted3 = classifier3.predict(X_test2)


# In[42]:


#Accuracy score 
accuracy3 = accuracy_score(y_test2, predicted3)
print(accuracy3)


# In[43]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier3, metrics.classification_report(y_test2, predicted3)))
disp = metrics.plot_confusion_matrix(classifier3, X_test2, y_test2)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[44]:


#utilizzo di RandomForestClassifier 
classifier4 = RandomForestClassifier(max_depth=15,max_features=2, random_state=0,n_estimators=20) 


# In[45]:


classifier4.fit(X_train2, y_train2)


# In[46]:


predicted4 = classifier4.predict(X_test2)


# In[47]:


#Accuracy score 
accuracy4= accuracy_score(y_test2, predicted4)
print(accuracy4)


# In[48]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier4, metrics.classification_report(y_test2, predicted4)))
disp = metrics.plot_confusion_matrix(classifier4, X_test2, y_test2)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[49]:


#Lavoriamo sul terzo dataset creato
Age_split3 = np.concatenate([df_Age1,df23,df_Age3])
Age_split3.shape


# In[50]:


df_age_split3 = pd.DataFrame(Age_split3,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split','Gender','trialsattempted'])
print(df_age_split3)


# In[51]:


df_test3 = df_age_split3[['MediaPP','MediaPR','MediaRR','MediaRP','Gender','trialsattempted']]
y3 = df_age_split3['Age_split']


# In[52]:


scaler = preprocessing.StandardScaler().fit(df_test3)
df_test3 = scaler.transform(df_test3)


# In[53]:


X_train3, X_test3, y_train3, y_test3 = train_test_split(df_test3, y3, test_size=0.3, random_state=0, shuffle = True)


# In[54]:


#utilizzo del classificazione svm.SVC
classifier5 = svm.SVC(C=1000,decision_function_shape='ovo',gamma='auto')#a support vector classifier


# In[55]:


classifier5.fit(X_train3, y_train3)


# In[56]:


predicted5 = classifier5.predict(X_test3)


# In[57]:


#Accuracy score 
accuracy5 = accuracy_score(y_test3, predicted5)
print(accuracy5)


# In[58]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier5, metrics.classification_report(y_test3, predicted5)))
disp = metrics.plot_confusion_matrix(classifier5, X_test3, y_test3)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[59]:


#utilizzo di RandomForestClassifier 
classifier6 = RandomForestClassifier(max_depth=15,max_features=2, random_state=42,n_estimators=10) 


# In[60]:


classifier6.fit(X_train3, y_train3)


# In[61]:


predicted6 = classifier6.predict(X_test3)


# In[62]:


#Accuracy score 
accuracy6 = accuracy_score(y_test3, predicted6)
print(accuracy6)


# In[63]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier6, metrics.classification_report(y_test3, predicted6)))
disp = metrics.plot_confusion_matrix(classifier6, X_test3, y_test3)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[64]:


#Calcolo media dell'accuracy dei 3 dataset per la SVM
accuracySVM = [accuracy1,accuracy3,accuracy5]
np.mean(accuracySVM)


# In[65]:


#Calcolo media dell'accuracy dei 3 dataset per RF
accuracyRF= [accuracy2,accuracy4,accuracy6]
np.mean(accuracyRF)


# In[ ]:




