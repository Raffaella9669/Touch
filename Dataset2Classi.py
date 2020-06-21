#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn import preprocessing




#In[2]:

# estrazione dati dal dataset
users = pd.read_excel('keystroke_51_Aggiornato.xls', 'users')
timers = pd.read_excel('keystroke_51_Aggiornato.xls', 'timers')
utenti = timers.join(users.set_index('UserName'), on='UserName')


# In[3]:


# Selezione delle colonne di interesse
df_utenti = pd.DataFrame(utenti, columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age'])
df_utenti.shape


# In[4]:


df_Age1 = df_utenti[df_utenti['Age'] == 1]
df_Age2 = df_utenti[df_utenti['Age'] == 2] 
df_Age3 = df_utenti[df_utenti['Age'] == 3]
#print(df_Age)


# In[5]:


#Divisione dataset per bilanciare la categoria 2 
df21 = df_Age2.iloc[:265]
df22 = df_Age2.iloc[265:530]


# In[6]:


#Lavoriamo sul primo dataset creato
Age_split = np.concatenate([df_Age1, df21, df_Age3])
Age_split.shape


# In[7]:


df_age_split = pd.DataFrame(Age_split, columns=['MediaPP', 'MediaPR', 'MediaRR', 'MediaRP', 'Age_split'])
print(df_age_split)


# In[8]:


df_age_split.info()


# In[9]:


#suddivide il dataset per età 
età = df_age_split['Age_split'].value_counts()
print(età)
explode = (0, 0,0.1) 
età.plot.pie(explode=explode, autopct='%1.1f%%', shadow=True, startangle=90, counterclock=False)


# In[10]:


#creazione del boxplot tra MediaPP e Age 
df_age_split.boxplot(column='MediaPP', by='Age_split', figsize=(8, 8))


# In[11]:


#creazione del boxplot tra MediaPR e Age 
df_age_split.boxplot(column='MediaPR', by='Age_split', figsize=(8, 8))


# In[12]:


#creazione del boxplot tra MediaRR e Age 
df_age_split.boxplot(column='MediaRR', by='Age_split', figsize=(8, 8))


# In[13]:


#creazione del boxplot tra MediaRP e Age 
df_age_split.boxplot(column='MediaRP', by='Age_split', figsize=(8, 8))


# In[14]:


df_test = df_age_split[['MediaPP', 'MediaPR', 'MediaRR', 'MediaRP']]
y = df_age_split['Age_split']


# In[15]:


print(y)


# In[16]:


print(df_test)


# In[17]:


df_test.info()


# In[18]:


scaler = preprocessing.StandardScaler().fit(df_test)
df_test = scaler.transform(df_test)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(df_test, y, test_size=0.3, random_state=42, shuffle=True)


# In[20]:


print(df_test.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[21]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per svm.SVC
parameters ={'C':[0.1, 1, 10, 100, 1000],
              'gamma':('auto','scale'),
              'kernel': ('linear', 'rbf', 'sigmoid'),
              'decision_function_shape':('ovo', 'ovr')
             }
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(df_test, y)
print("Classifier 1:", clf.best_estimator_)


# In[22]:


#Evaluate a score by cross-validation
cv_results = model_selection.cross_val_score(clf, df_test, y, cv=3)
print(cv_results.mean())


# In[23]:


#utilizzo del classificazione svm.SVC
classifier = svm.SVC(C=10, decision_function_shape='ovo', gamma='scale')#a support vector classifier


# In[24]:


classifier.fit(X_train, y_train)


# In[25]:


predicted = classifier.predict(X_test)


# In[26]:


predicted.shape
y_test.shape


# In[27]:


print(y_test.values)


# In[28]:


print(predicted)


# In[29]:


#Accuracy score 
accuracy1 = accuracy_score(y_test, predicted)
print("Accuracy 1: ", accuracy1)


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
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf = RandomForestClassifier()
clf2 = GridSearchCV(rf, parameters2)
clf2.fit(df_test, y)
print("Classifier 2:", clf2.best_estimator_)


# In[32]:


clf2.best_score_


# In[33]:


#Evaluate a score by cross-validation
cv_results2 = model_selection.cross_val_score(clf2, df_test, y)
print(cv_results2.mean())


# In[34]:


#utilizzo di RandomForestClassifier 
classifier2 = RandomForestClassifier(max_depth=5, max_features=4, random_state=42, n_estimators=10)


# In[35]:


classifier2.fit(X_train, y_train)


# In[36]:


predicted2 = classifier2.predict(X_test)


# In[37]:


#Accuracy score 
accuracy2 = accuracy_score(y_test, predicted2)
print("Accuracy 2:", accuracy2)


# In[38]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier2, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier2, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[20]:


#Lavoriamo sul secondo dataset creato
Age_split2 = np.concatenate([df_Age1,df22,df_Age3])
Age_split2.shape


# In[21]:


df_age_split2 = pd.DataFrame(Age_split2,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split'])
print(df_age_split2)


# In[22]:


df_test2 = df_age_split2[['MediaPP','MediaPR','MediaRR','MediaRP']]
y2 = df_age_split2['Age_split']


# In[23]:


scaler = preprocessing.StandardScaler().fit(df_test2)
df_test2 = scaler.transform(df_test2)


# In[24]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(df_test2, y2, test_size=0.3, random_state=0, shuffle=True)


# In[44]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per svm.SVC
parameters3 ={'C':[0.1, 1, 10, 100, 1000],
              'gamma':('auto', 'scale'),
              'kernel': ('linear', 'rbf', 'sigmoid'),
              'decision_function_shape':('ovo', 'ovr')
             }
svc2 = svm.SVC()
clf3 = GridSearchCV(svc2, parameters3)
clf3.fit(df_test2, y2)
print("Classifier 3:", clf3.best_estimator_)


# In[45]:


clf3.best_score_


# In[46]:


#Evaluate a score by cross-validation
cv_results3 = model_selection.cross_val_score(clf3, df_test2, y2)
print(cv_results3.mean())


# In[47]:


#utilizzo del classificazione svm.SVC
classifier3 = svm.SVC(C=1000, decision_function_shape='ovo', gamma='scale')#a support vector classifier


# In[48]:


classifier3.fit(X_train2, y_train2)


# In[49]:


predicted3 = classifier3.predict(X_test2)


# In[50]:


#Accuracy score 
accuracy3 = accuracy_score(y_test2, predicted3)
print("Accuracy 3:", accuracy3)


# In[51]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier3, metrics.classification_report(y_test2, predicted3)))
disp = metrics.plot_confusion_matrix(classifier3, X_test2, y_test2)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[52]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per RandomForestClassifier
parameters4 = {'n_estimators': [10, 20, 50, 100],
              'bootstrap': [True, False],
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf2 = RandomForestClassifier()
clf4 = GridSearchCV(rf2, parameters4)
clf4.fit(df_test2, y2)
print("Classifier4: ", clf4.best_estimator_)


# In[53]:


clf4.best_score_


# In[54]:


#Evaluate a score by cross-validation
cv_results4 = model_selection.cross_val_score(clf4, df_test2, y2)
print(cv_results4.mean())


# In[25]:


#utilizzo di RandomForestClassifier 
classifier4 = RandomForestClassifier(max_depth=5, max_features=4, random_state=42, n_estimators=50)


# In[26]:


classifier4.fit(X_train2, y_train2)


# In[27]:


predicted4 = classifier4.predict(X_test2)


# In[28]:


#Accuracy score 
accuracy4 = accuracy_score(y_test2, predicted4)
print("Accuracy 4:", accuracy4)


# In[30]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier4, metrics.classification_report(y_test2, predicted4)))
disp = metrics.plot_confusion_matrix(classifier4, X_test2, y_test2)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[63]:


#Calcolo media dell'accuracy dei 2 dataset per la SVM
accuracySVM = [accuracy1, accuracy3]
print("Accuracy SVM:", np.mean(accuracySVM))


# In[64]:


#Calcolo media dell'accuracy dei 2 dataset per RF
accuracyRF = [accuracy2, accuracy4]
print("Accuracy RF:", np.mean(accuracyRF))






