import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Lettura Excel del foglio users.
users = pd.read_excel('keystroke_51_Aggiornato.xls', 'users')

print(users.columns)

#per filtrare i dati utilizziamo la funzione iloc
#il primo parametro sono le righe, mentre il secondo rappresenta la classe Age
Y = users.iloc[:, 4].values
#print (Y)

#Trasformiamo l'array in dataframe
dataFrame_Y= pd.DataFrame(Y)
#print(dataFrame_Y)

#Seleziono utenti con età compresa tra 7-18 anni
piccolo = dataFrame_Y[(dataFrame_Y == 1)].dropna()
#cast forzato ad int
#print(piccolo.astype(int))

#Seleziono utenti con età compresa tra 19-29 anni
medio=dataFrame_Y[(dataFrame_Y==2)].dropna()
#cast forzato ad int
#print(medio.astype(int))

#Seleziono utenti con età compresa tra 30-65 anni
grande = dataFrame_Y[(dataFrame_Y==3)].dropna()
#print(grande.astype(int))

#creazione labels per il grafico
labels = '7-18 Anni', '19-29 Anni', '30-65 Anni'
print("Utenti con età compresa tra 7-18 anni:",piccolo.size)
print("Utenti con età compresa tra 19-29 anni:",medio.size)
print("Utenti con età compresa tra 30-65 anni:",grande.size)
#assegno alla variabile size il valore delle lables
sizes = [piccolo.size, medio.size, grande.size]
explode = (0, 0,0.1)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, counterclock=False)
fig1.suptitle('Suddivisione utenti per età')
ax1.axis('equal')

plt.show()
