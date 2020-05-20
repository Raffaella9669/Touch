import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


users = pd.read_excel('keystroke_51_Aggiornato.xls', 'users')
timers = pd.read_excel('keystroke_51_Aggiornato.xls', 'timers')

utenti = timers.join(users.set_index('UserName'), on='UserName')
#print(utenti.columns)
print(utenti['Age'].value_counts())
print(utenti)

sns.boxplot(x='Age', y='SommaPP', data=utenti)

plt.show()
sns.boxplot(x='Age', y='SommaPR',data=utenti)
plt.show()

sns.boxplot(x='Age', y='SommaRR',data=utenti)
plt.show()

sns.boxplot(x='Age', y='SommaRP',data=utenti)
plt.show()



