URL = "https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD"

import pandas as pd
import matplotlib.pyplot as plt

local_file = "data/fremont.csv"

##chargement 
data = pd.read_csv(local_file)
data.shape

#print(data.head())

##doublons

#Exercice
inplace=True #pour que les modifications soient faites directement sur le tableau d'orgine
#print(data.shape[:2])
data=data.drop_duplicates()
#print(data.shape[:2])

##échauffement
sample_dates = pd.to_datetime([
    "2024-01-15 08:00:00",
    "2024-01-20 17:30:00",
    "2024-02-03 12:00:00",
])

#1.
#print(type(sample_dates))
#le type de sample_dates est pandas.DatetimeIndex

#2.
#print(sample_dates.date) # retourne la partie date
#print(sample_dates.time) # retourne la partie horaire 
#print(sample_dates.dayofweek) # retourne Index([0, 5, 5], dtype='int32') qui correspond aux jours de la semaine (0 pour lundi et 6 pour dimanche)

#3. on a déjà que des dates postérieures ou égales au 20 janvier 2024

#4. 
print(sample_dates[sample_dates<'2024-02-01']) # on appel ce qui y a entre crochet (la condition à vérifer) le "mask"

##parser les dates
# intéressant aussi, pour voir notamment les points manquants
data.info()
data.dtypes

data.index=pd.to_datetime(data.Date, format= "%m/%d/%Y %I:%M:%S %p") #on modifie l'index de notre tableau
data.drop('Date',axis=1,inplace=True)
#renommons les colonnes
data.columns = ['Total', 'West', 'East']

##données manquantes et extension types
#Exercice
#1.
#print(data[data.isna()==True])
#2.
data=data.convert_dtypes()

# un premier jet, pas terrible du tout
data[['East', 'West']].plot()

##resample(): changer la fréquence temporelle
#Exercice
#1.
data.resample("1W").sum().plot() 
#on peut faire un plt.figsize((..,..))
plt.xlabel("semaines")
plt.ylabel("nb de passages")
#plt.show()

#2.
print(len(data)/len(data.resample("1W").sum())) #affiche 167,8 ce qui correspond bien à 7*24=168

##découpage temporel
#Exercice
data=data[data.index<'2017-01-01']

##évolution annuelle
#Exercice
data.resample("1YE").sum().plot()
plt.xlabel('années')
plt.ylabel('nb de passages')
#plt.show()
#rmq: 2012 n'est pas comparable aux autres années car on a les donées que pr qq mois de 2012

##profil journalier moyen
data.groupby(data.index.time).mean().plot()
plt.xlabel('heures')
plt.ylabel('nb de passages')
plt.show()

##une courbe par jour : la pivot_table
pivoted=data.pivot_table(values='Total', index=data.index.time, columns=data.index.date)
pivoted.plot(legend=False, alpha=0.01)
plt.xlabel('heures')
plt.ylabel('nb de passages')
#plt.show()


##clasification des jours
# .T pour transposer et .fillna(0) pour remplir les valeurs manquantes par 0
pca_input = pivoted.T.fillna(0) 

###ACP
from sklearn.decomposition import PCA
# on utilise le PCA de scikit-learn comme une boite noire
pca_output = PCA(2).fit_transform(pca_input) #on réduit la dimension du tableau à 2
print('dimension=',pca_output.shape) #on a bien 2 colonnes
#on affiche un graphique test sans utiliser le modèle gaussien
plt.scatter(pca_output[:, 0], pca_output[:, 1], marker='.', color='blue')
plt.title("graphique test")
plt.show()

### GaussianMixture
#Exercice
#on utilise maintenant un modèle de mélange gaussien pour trouver ces 2 groupes
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(2)
labels = gmm.fit(pca_input).predict(pca_input) #on attribue la note à pour le 1er cluster, et 1 pour le 2e
plt.scatter(pca_output[:,0],pca_output[:,1],c=labels,cmap='viridis')
plt.title("modèle gaussien")
plt.show()

##Comparons chaque famille aux profils journaliers
#Exercice
plt.plot(pca_input[labels==0].mean(axis=0).values, label="jours de la semaine", color='blue') #.values sert à convertir un objet Pandas en un tableau de nombres lisible par la fonction plt.plot
plt.plot(pca_input[labels==1].mean(axis=0).values, label='week-end', color='magenta')
plt.xlabel('heures')
plt.ylabel("nb de passages")
plt.legend()
plt.show()
#on observe 2 pics pour les jours de lasemaine (du lundi au vendredi) correspondant au moment ou les gens vont et rentrent du travail
#pour les jours du weekends ont voient que le nomnbre de passage est important surtout entre 10h et 16h (promenades)

##Vérifions avec le jour de la semaine
#Exercice
plt.scatter(pca_output[:,0],pca_output[:,1], c=pd.to_datetime(pivoted.columns).dayofweek, cmap='tab10', marker='.') #cmap va nous donner une palette de couleurs
plt.title('couleurs en fonction du jour de la semaine')
plt.show()

##Les moutons noirs
moutons_noirs=pivoted.loc[:, (pd.to_datetime(pivoted.columns).dayofweek<5) & (labels==1)]
plt.plot(moutons_noirs.mean(axis=1).values, color='black') 
plt.title('les moutons noirs')
plt.show()
#le profil des moutons noirs correspondent au profil des jours du cluster 0