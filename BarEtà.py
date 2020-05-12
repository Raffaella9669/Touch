import matplotlib.pyplot as plt


from SuddivisioneDataSet import piccolo, medio, grande,labels

labels = labels
p=piccolo.size
m=medio.size
g=grande.size
y = [p, m, g]

fig, ax2 = plt.subplots()

xticks = [1,2,3] # ci serve per posizionare le barre e anche le label

ax2.bar(xticks, y, align='center')
ax2.set_title("Divisone per età")
ax2.set_xticklabels(labels)  # verranno posizionate dove sono gli xticks
ax2.set_xticks(xticks)

plt.show()



