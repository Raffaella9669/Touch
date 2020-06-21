#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#Lettura Excel del foglio users.
users = pd.read_excel('keystroke_51_Aggiornato.xls','users')
#users
users.shape


# In[3]:


users.columns


# In[4]:


#per filtrare i dati utilizziamo la funzione iloc 
#il primo parametro sono le righe, mentre il secondo rappresenta la classe Age 
Y = users.iloc[:, 4].values 
print (Y)


# In[5]:


#Trasformiamo l'array in dataframe
dataFrame_Y= pd.DataFrame(Y)
dataFrame_Y


# In[6]:


#Seleziono utenti con età compresa tra 7-18 anni 
piccolo = dataFrame_Y[(dataFrame_Y == 1)].dropna()
#cast forzato ad int
piccolo.astype(int)


# In[7]:


#Seleziono utenti con età compresa tra 19-29 anni 
medio=dataFrame_Y[(dataFrame_Y==2)].dropna()
#cast forzato ad int 
medio.astype(int)


# In[8]:


#Seleziono utenti con età compresa tra 30-65 anni 
grande = dataFrame_Y[(dataFrame_Y==3)].dropna()
grande.astype(int)


# In[9]:


#visualizziamo tramite l'utilizzo di grafici, i dati filtrati.
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


#creazione labels per il grafico
labels = '7-18 Anni', '19-29 Anni', '30-65 Anni'
#print("Utenti con età compresa tra 7-18 anni:",piccolo.size)
#print("Utenti con età compresa tra 19-29 anni:",medio.size)
#print("Utenti con età compresa tra 30-65 anni:",grande.size)
#assegno alla variabile size il valore delle lables
sizes = [piccolo.size, medio.size, grande.size]
explode = (0, 0,0.1)  

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, counterclock=False)
fig1.suptitle('Suddivisione utenti per età')
ax1.axis('equal') 
plt.show()


# In[11]:


labels = labels
p=piccolo.size
m=medio.size
g=grande.size
y = [p, m, g]

fig, ax2 = plt.subplots()

xticks = [1,2,3] # ci serve per posizionare le barre e anche le label

ax2.bar(xticks, y, align='center')
ax2.set_title("Suddivisione utenti per età")
ax2.set_xticklabels(labels)  # verranno posizionate dove sono gli xticks
ax2.set_xticks(xticks)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




