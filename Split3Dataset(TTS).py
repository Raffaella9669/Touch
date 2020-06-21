#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
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


# In[15]:


y


# In[16]:


df_test.info()


# In[18]:


#Accuracy 0.64
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


#Accuracy 0.63
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(df_test)
df_test = transformer.transform(df_test)
df_test


# In[19]:


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


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per svm.SVC
parameters =  {'C':[0.1,1,10,100,1000],
              'gamma':('auto','scale'),
              'kernel': ('linear','rbf','sigmoid'),
              'decision_function_shape':('ovo','ovr')
             }
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(df_test,y)
clf.best_estimator_


# In[23]:


#Evaluate a score by cross-validation
cv_results = model_selection.cross_val_score(clf,df_test,y) 
print (cv_results.mean())


# In[24]:


#utilizzo del classificazione svm.SVC
classifier = svm.SVC(C=10,decision_function_shape='ovo',gamma='auto')


# In[25]:


classifier.fit(X_train, y_train)


# In[26]:


predicted = classifier.predict(X_test)


# In[27]:


predicted.shape
y_test.shape


# In[28]:


print(predicted)


# In[29]:


#Accuracy score 
accuracy1 = accuracy_score(y_test, predicted)
print(accuracy1)


# In[30]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[31]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per RandomForestClassifier
parameters2 = {'n_estimators': [10, 20, 50, 100],
              'bootstrap': [True, False],
              'criterion': ["gini", "entropy"],
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf = RandomForestClassifier()
clf2 = GridSearchCV(rf, parameters2)
clf2.fit(df_test,y)
clf2.best_estimator_


# In[32]:


#Evaluate a score by cross-validation
cv_results2 = model_selection.cross_val_score(clf2,df_test,y) 
print (cv_results2.mean())


# In[33]:


#utilizzo di RandomForestClassifier 
classifier2 = RandomForestClassifier(max_depth=5,max_features=4, random_state=0,n_estimators=100) 


# In[34]:


classifier2.fit(X_train, y_train)


# In[35]:


predicted2 = classifier2.predict(X_test)


# In[36]:


#Accuracy score 
accuracy2 = accuracy_score(y_test, predicted2)
print(accuracy2)


# In[37]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier2, metrics.classification_report(y_test, predicted2)))
disp = metrics.plot_confusion_matrix(classifier2, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[38]:


#Lavoriamo sul secondo dataset creato
Age_split2 = np.concatenate([df_Age1,df22,df_Age3])
Age_split2.shape


# In[39]:


df_age_split2 = pd.DataFrame(Age_split2,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split'])
print(df_age_split2)


# In[40]:


df_test2 = df_age_split2[['MediaPP','MediaPR','MediaRR','MediaRP']]
y2 = df_age_split2['Age_split']


# In[41]:


scaler = preprocessing.StandardScaler().fit(df_test2)
df_test2 = scaler.transform(df_test2)


# In[42]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(df_test2, y2, test_size=0.3, random_state=0, shuffle = True )


# In[43]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per svm.SVC
parameters3 =  {'C':[0.1,1,10,100,1000],
              'gamma':('auto','scale'),
              'kernel': ('linear','rbf','sigmoid'),
              'decision_function_shape':('ovo','ovr')
             }
svc2 = svm.SVC()
clf3 = GridSearchCV(svc2, parameters3)
clf3.fit(df_test2,y2)
clf3.best_estimator_


# In[44]:


#Evaluate a score by cross-validation
cv_results3 = model_selection.cross_val_score(clf3,df_test2,y2) 
print (cv_results3.mean())


# In[45]:


classifier3 = svm.SVC(C=100,decision_function_shape='ovo',gamma='auto')#a support vector classifier


# In[46]:


classifier3.fit(X_train2, y_train2)


# In[47]:


predicted3 = classifier3.predict(X_test2)


# In[48]:


#Accuracy score 
accuracy3 = accuracy_score(y_test2, predicted3)
print(accuracy3)


# In[49]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier3, metrics.classification_report(y_test2, predicted3)))
disp = metrics.plot_confusion_matrix(classifier3, X_test2, y_test2)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[50]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per RandomForestClassifier
parameters4 = {'n_estimators': [10, 20, 50, 100],
              'bootstrap': [True, False],
              'criterion': ["gini", "entropy"],
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf2 = RandomForestClassifier()
clf4 = GridSearchCV(rf2, parameters4)
clf4.fit(df_test2,y2)
clf4.best_estimator_


# In[51]:


#Evaluate a score by cross-validation
cv_results4 = model_selection.cross_val_score(clf4,df_test2,y2) 
print (cv_results4.mean())


# In[52]:


#utilizzo di RandomForestClassifier 
classifier4 = RandomForestClassifier(max_depth=5,max_features=4, random_state=0,n_estimators=20) 


# In[53]:


classifier4.fit(X_train2, y_train2)


# In[54]:


predicted4 = classifier4.predict(X_test2)


# In[55]:


#Accuracy score 
accuracy4= accuracy_score(y_test2, predicted4)
print(accuracy4)


# In[56]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier4, metrics.classification_report(y_test2, predicted4)))
disp = metrics.plot_confusion_matrix(classifier3, X_test2, y_test2)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[57]:


#Lavoriamo sul terzo dataset creato
Age_split3 = np.concatenate([df_Age1,df23,df_Age3])
Age_split3.shape


# In[58]:


df_age_split3 = pd.DataFrame(Age_split3,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split'])
print(df_age_split3)


# In[59]:


df_test3 = df_age_split3[['MediaPP','MediaPR','MediaRR','MediaRP']]
y3 = df_age_split3['Age_split']


# In[60]:


scaler = preprocessing.StandardScaler().fit(df_test3)
df_test3 = scaler.transform(df_test3)


# In[61]:


X_train3, X_test3, y_train3, y_test3 = train_test_split(df_test3, y3, test_size=0.3, random_state=0, shuffle = True)


# In[62]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per svm.SVC
parameters5 = {'C':[0.1,1,10,100,1000],
              'gamma':('auto','scale'),
              'kernel': ('linear','rbf','sigmoid'),
              'decision_function_shape':('ovo','ovr')
             }
svc3 = svm.SVC()
clf5 = GridSearchCV(svc3, parameters5)
clf5.fit(df_test3,y3)
clf5.best_estimator_


# In[63]:


#Evaluate a score by cross-validation
cv_results5 = model_selection.cross_val_score(clf5,df_test3,y3) 
print (cv_results5.mean())


# In[64]:


#utilizzo del classificazione svm.SVC
classifier5 = svm.SVC(C=10,decision_function_shape='ovo',gamma='scale')#a support vector classifier


# In[65]:


classifier5.fit(X_train3, y_train3)


# In[66]:


predicted5 = classifier5.predict(X_test3)


# In[67]:


#Accuracy score 
accuracy5 = accuracy_score(y_test3, predicted5)
print(accuracy5)


# In[68]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier5, metrics.classification_report(y_test3, predicted5)))
disp = metrics.plot_confusion_matrix(classifier5, X_test3, y_test3)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[69]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per RandomForestClassifier
parameters6 = {'n_estimators': [10, 20, 50, 100],
              'bootstrap': [True, False],
              'criterion': ["gini", "entropy"],
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf3 = RandomForestClassifier()
clf6 = GridSearchCV(rf3, parameters6)
clf6.fit(df_test3,y3)
clf6.best_estimator_


# In[70]:


#Evaluate a score by cross-validation
cv_results6 = model_selection.cross_val_score(clf6,df_test3,y3) 
print (cv_results6.mean())


# In[71]:


#utilizzo di RandomForestClassifier 
classifier6 = RandomForestClassifier(criterion = 'entropy', max_depth=10,max_features=2, random_state=0,n_estimators=20) 


# In[72]:


classifier6.fit(X_train3, y_train3)


# In[73]:


predicted6 = classifier6.predict(X_test3)


# In[74]:


#Accuracy score 
accuracy6 = accuracy_score(y_test3, predicted6)
print(accuracy6)


# In[75]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier6, metrics.classification_report(y_test3, predicted6)))
disp = metrics.plot_confusion_matrix(classifier6, X_test3, y_test3)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[76]:


#Calcolo media dell'accuracy dei 3 dataset per la SVM
accuracySVM = [accuracy1,accuracy3,accuracy5]
np.mean(accuracySVM)


# In[77]:


#Calcolo media dell'accuracy dei 3 dataset per RF
accuracyRF= [accuracy2,accuracy4,accuracy6]
np.mean(accuracyRF)


# In[ ]:




