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
from sklearn import model_selection
from numpy import mean



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


df_test = df_age_split[['MediaPP','MediaPR','MediaRR','MediaRP','Gender','trialsattempted']]
y = df_age_split['Age_split']


# In[15]:


y


# In[16]:


df_test.info()


# In[ ]:


sns.pairplot(df_age_split, hue = 'Age_split', palette = 'Dark2')


# In[17]:


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


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(df_test, y, test_size=0.3, shuffle = True, random_state = 42)


# In[19]:


df_test


# In[20]:


print(df_test.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[21]:


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
classifier = svm.SVC(C=10,decision_function_shape='ovo',gamma='scale')


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


# In[46]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per RandomForestClassifier
parameters2 = {'n_estimators': [10, 20, 50, 100],
              'bootstrap': [True, False],
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf = RandomForestClassifier()
clf2 = GridSearchCV(rf, parameters2)
clf2.fit(df_test,y)
clf2.best_estimator_


# In[47]:


clf2.best_score_


# In[48]:


#Evaluate a score by cross-validation
cv_results2 = model_selection.cross_val_score(clf2,df_test,y) 
print (cv_results2.mean())


# In[49]:


#utilizzo di RandomForestClassifier 
classifier2 = RandomForestClassifier(max_depth=10,max_features=4, random_state=42,n_estimators=20) 


# In[50]:


classifier2.fit(X_train, y_train)


# In[51]:


predicted2 = classifier2.predict(X_test)


# In[52]:


#Accuracy score 
accuracy2 = accuracy_score(y_test, predicted2)
print(accuracy2)


# In[53]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier2, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier2, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[54]:


#Lavoriamo sul secondo dataset creato
Age_split2 = np.concatenate([df_Age1,df22,df_Age3])
Age_split2.shape


# In[55]:


df_age_split2 = pd.DataFrame(Age_split2,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split','Gender','trialsattempted'])
print(df_age_split2)


# In[56]:


df_test2 = df_age_split2[['MediaPP','MediaPR','MediaRR','MediaRP','Gender','trialsattempted']]
y2 = df_age_split2['Age_split']


# In[57]:


scaler = preprocessing.StandardScaler().fit(df_test2)
df_test2 = scaler.transform(df_test2)


# In[58]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(df_test2, y2, test_size=0.3, random_state=0, shuffle = True )


# In[59]:


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


# In[60]:


clf3.best_score_


# In[61]:


#Evaluate a score by cross-validation
cv_results3 = model_selection.cross_val_score(clf3,df_test2,y2) 
print (cv_results3.mean())


# In[62]:


classifier3 = svm.SVC(C=10,decision_function_shape='ovo',gamma='auto')#a support vector classifier


# In[63]:


classifier3.fit(X_train2, y_train2)


# In[64]:


predicted3 = classifier3.predict(X_test2)


# In[65]:


#Accuracy score 
accuracy3 = accuracy_score(y_test2, predicted3)
print(accuracy3)


# In[66]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier3, metrics.classification_report(y_test2, predicted3)))
disp = metrics.plot_confusion_matrix(classifier3, X_test2, y_test2)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[68]:


#Utilizzo di GridSearchCV per la ricerca dei valori migliori per RandomForestClassifier
parameters4 = {'n_estimators': [10, 20, 50, 100],
              'bootstrap': [True, False],
              'max_depth':[5,10,15],
              'max_features':[2,4],
              'random_state':[0,42]
             }
rf2 = RandomForestClassifier()
clf4 = GridSearchCV(rf2, parameters4)
clf4.fit(df_test2,y2)
clf4.best_estimator_


# In[69]:


clf4.best_score_


# In[70]:


#Evaluate a score by cross-validation
cv_results4 = model_selection.cross_val_score(clf4,df_test2,y2) 
print (cv_results4.mean())


# In[71]:


#utilizzo di RandomForestClassifier 
classifier4 = RandomForestClassifier(max_depth=15,max_features=2, random_state=0,n_estimators=20) 


# In[72]:


classifier4.fit(X_train2, y_train2)


# In[73]:


predicted4 = classifier4.predict(X_test2)


# In[74]:


#Accuracy score 
accuracy4= accuracy_score(y_test2, predicted4)
print(accuracy4)


# In[75]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier3, metrics.classification_report(y_test2, predicted3)))
disp = metrics.plot_confusion_matrix(classifier4, X_test2, y_test2)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[76]:


#Lavoriamo sul terzo dataset creato
Age_split3 = np.concatenate([df_Age1,df23,df_Age3])
Age_split3.shape


# In[77]:


df_age_split3 = pd.DataFrame(Age_split3,columns = ['MediaPP','MediaPR','MediaRR','MediaRP','Age_split','Gender','trialsattempted'])
print(df_age_split3)


# In[78]:


df_test3 = df_age_split3[['MediaPP','MediaPR','MediaRR','MediaRP','Gender','trialsattempted']]
y3 = df_age_split3['Age_split']


# In[79]:


scaler = preprocessing.StandardScaler().fit(df_test3)
df_test3 = scaler.transform(df_test3)


# In[80]:


X_train3, X_test3, y_train3, y_test3 = train_test_split(df_test3, y3, test_size=0.3, random_state=0, shuffle = True)


# In[81]:


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


# In[82]:


clf5.best_score_


# In[83]:


#Evaluate a score by cross-validation
cv_results5 = model_selection.cross_val_score(clf5,df_test3,y3) 
print (cv_results5.mean())


# In[84]:


#utilizzo del classificazione svm.SVC
classifier5 = svm.SVC(C=1000,decision_function_shape='ovo',gamma='auto')#a support vector classifier


# In[85]:


classifier5.fit(X_train3, y_train3)


# In[86]:


predicted5 = classifier5.predict(X_test3)


# In[87]:


#Accuracy score 
accuracy5 = accuracy_score(y_test3, predicted5)
print(accuracy5)


# In[88]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier5, metrics.classification_report(y_test3, predicted5)))
disp = metrics.plot_confusion_matrix(classifier5, X_test3, y_test3)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[89]:


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


# In[90]:


clf6.best_score_


# In[91]:


#Evaluate a score by cross-validation
cv_results6 = model_selection.cross_val_score(clf6,df_test3,y3) 
print (cv_results6.mean())


# In[92]:


#utilizzo di RandomForestClassifier 
classifier6 = RandomForestClassifier(max_depth=15,max_features=2, random_state=42,n_estimators=10) 


# In[93]:


classifier6.fit(X_train3, y_train3)


# In[94]:


predicted6 = classifier6.predict(X_test3)


# In[95]:


#Accuracy score 
accuracy6 = accuracy_score(y_test3, predicted6)
print(accuracy6)


# In[96]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier6, metrics.classification_report(y_test3, predicted6)))
disp = metrics.plot_confusion_matrix(classifier6, X_test3, y_test3)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[97]:


#Calcolo media dell'accuracy dei 3 dataset per la SVM
accuracySVM = [accuracy1,accuracy3,accuracy5]
np.mean(accuracySVM)


# In[98]:


#Calcolo media dell'accuracy dei 3 dataset per RF
accuracyRF= [accuracy2,accuracy4,accuracy6]
np.mean(accuracyRF)


# In[ ]:




