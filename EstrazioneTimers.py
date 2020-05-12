import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics


timers = pd.read_excel('keystroke_51.xls', 'timers')

print(timers.columns)
print(timers.shape)


valori = timers.iloc[:, 3:7].values
#print(valori)
print(type(valori))
df_valori= pd.DataFrame(valori)
#print(df_valori)
PP = df_valori.iloc[:, 0:1].values
df_PP = pd.DataFrame(PP)
print(df_PP)






