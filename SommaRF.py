#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn import model_selection
from sklearn.metrics import accuracy_score


# In[2]:


#estrazione dati dal dataset 
users = pd.read_excel('keystroke_51_Aggiornato.xls', 'users')
timers = pd.read_excel('keystroke_51_Aggiornato.xls', 'timers')
utenti = timers.join(users.set_index('UserName'), on='UserName')


# In[3]:


utenti.shape


# In[4]:


#Selezione delle colonne di interesse 
df_utenti = pd.DataFrame(utenti,columns = ['SommaPP','SommaPR','SommaRR','SommaRP','Age'])
df_utenti.shape


# In[5]:


df_utenti.mean()


# In[6]:


df_utenti.columns


# In[7]:


df_utenti.info()


# In[8]:


#suddivide il dataset per età 
età=df_utenti['Age'].value_counts()
print(età)
explode = (0, 0,0.1) 
età.plot.pie(explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90, counterclock=False)


# In[9]:


#Utilizzo della libreria seaborn per la creazione del boxplot tra SommaPP e Age
sns.boxplot(x='Age', y='SommaPP', data=df_utenti)


# In[10]:


#Utilizzo della libreria seaborn per la creazione del boxplot tra SommaPR e Age
sns.boxplot(x='Age', y='SommaPR',data=df_utenti)


# In[11]:


#Utilizzo della libreria seaborn per la creazione del boxplot tra SommaRR e Age
sns.boxplot(x='Age', y='SommaRR',data=df_utenti)


# In[12]:


#Utilizzo della libreria seaborn per la creazione del boxplot tra SommaRP e Age
sns.boxplot(x='Age', y='SommaRP',data=df_utenti)


# In[13]:


#creazione del boxplot tra SommaPP e Age 
df_utenti.boxplot(column='SommaPP', by='Age',figsize=(8,8))


# In[14]:


#creazione del boxplot tra SommaPR e Age 
df_utenti.boxplot(column='SommaPR', by='Age',figsize=(8,8))


# In[15]:


#creazione del boxplot tra SommaRR e Age 
df_utenti.boxplot(column='SommaRR', by='Age',figsize=(8,8))


# In[16]:


#creazione del boxplot tra SommaRP e Age 
df_utenti.boxplot(column='SommaRP', by='Age',figsize=(8,8))


# In[17]:


df_utenti.columns


# In[18]:


#Calcolo della media della Somma delle 4 Features analizzate raggruppando per Età
count1=df_utenti[['Age','SommaPP']].groupby(['Age'],as_index=False).mean().sort_values(by='Age',ascending=False)
print(count1)
print('-'*40)
count2=df_utenti[['Age','SommaPR']].groupby(['Age'],as_index=False).mean().sort_values(by='Age',ascending=False)
print(count2)
print('-'*40)
count3=df_utenti[['Age','SommaRR']].groupby(['Age'],as_index=False).mean().sort_values(by='Age',ascending=False)
print(count3)
print('-'*40)
count4=df_utenti[['Age','SommaRP']].groupby(['Age'],as_index=False).mean().sort_values(by='Age',ascending=False)
print(count4)


# In[19]:


#Selezione valori della media per la creazione dei grafici 
x3 = ((count1.SommaPP.values[0]))
x2 = ((count1.SommaPP.values[1]))
x1 = ((count1.SommaPP.values[2]))


# In[20]:


#Creazione plot tra la media di tutte le sommePP e Age
x = ['7-18','19-29','30-60'] # Crea un array dei valori x 
y = [x1,x2,x3]  # Crea un array dei corrispondenti valori y
plt.plot(x, y,'g') 
plt.plot(x,y,'go')
# Usa pylab per tracciare con  x,y
plt.grid()
plt.show()


# In[21]:


#Selezione valori della media per la creazione dei grafici 
x3 = ((count2.SommaPR.values[0]))
x2 = ((count2.SommaPR.values[1]))
x1 = ((count2.SommaPR.values[2]))


# In[22]:


#Creazione plot tra la media di tutte le sommePR e Age
x = ['7-18','19-29','30-60'] # Crea un array dei valori x 
y = [x1,x2,x3]  # Crea un array dei corrispondenti valori y
plt.plot(x, y,'g') 
plt.plot(x,y,'go')
# Usa pylab per tracciare con  x,y
plt.grid()
plt.show()


# In[23]:


#Selezione valori della media per la creazione dei grafici 
x3 = ((count3.SommaRR.values[0]))
x2 = ((count3.SommaRR.values[1]))
x1 = ((count3.SommaRR.values[2]))


# In[24]:


#Creazione plot tra la media di tutte le sommeRR e Age
x = ['7-18','19-29','30-60'] # Crea un array dei valori x 
y = [x1,x2,x3]  # Crea un array dei corrispondenti valori y
plt.plot(x, y,'g') 
plt.plot(x,y,'go')
# Usa pylab per tracciare con  x,y
plt.grid()
plt.show()


# In[25]:


#Selezione valori della media per la creazione dei grafici 
x3 = ((count4.SommaRP.values[0]))
x2 = ((count4.SommaRP.values[1]))
x1 = ((count4.SommaRP.values[2]))


# In[26]:


#Creazione plot tra la media di tutte le sommeRP e Age
x = ['7-18','19-29','30-60'] # Crea un array dei valori x 
y = [x1,x2,x3]  # Crea un array dei corrispondenti valori y
plt.plot(x, y,'g') 
plt.plot(x,y,'go')
# Usa pylab per tracciare con  x,y
plt.grid()
plt.show()


# In[27]:


df_utenti.shape


# In[28]:


df_utenti = df_utenti.fillna(method='ffill')
#utenti = utenti.fillna(method= 'ffill')


# In[29]:


df_test = pd.DataFrame(utenti,columns = ['SommaPP','SommaPR','SommaRR','SommaRP'])
y = df_utenti['Age']


# In[30]:


df_utenti.info()


# In[31]:


scaler = StandardScaler()
df_test = scaler.fit_transform(df_test)


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(df_test, y, test_size=0.3, random_state = 42, shuffle = True )


# In[33]:


print(df_test.shape)
print(df_utenti.Age.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[34]:


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


# In[42]:


#Evaluate a score by cross-validation
cv_results = model_selection.cross_val_score(clf,df_test,y) 
print (cv_results.mean())


# In[35]:


classifier = RandomForestClassifier(max_depth=5, n_estimators=100,random_state=42,criterion='entropy',max_features=4) 


# In[36]:


classifier.fit(X_train, y_train)


# In[37]:


predicted = classifier.predict(X_test)


# In[38]:


predicted.shape
y_test.shape


# In[39]:


print(predicted)


# In[40]:


#Accuracy score 
print(accuracy_score(y_test, predicted))


# In[41]:


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)


# In[ ]:




