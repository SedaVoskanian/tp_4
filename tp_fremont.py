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
#print(sample_dates.dayofweek) # retourne Index([0, 5, 5], dtype='int32') qui correspond aux jours de la semaine (0 pour lundi et 7 pour dimanche)

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
plt.show()


##clasification des jours
# on transpose, et on remplit les valeurs manquantes par 0
pca_input = pivoted.T.fillna(0)

###ACP
from sklearn.decomposition import PCA
# on utilise le PCA de scikit-learn comme une boite noire
pca_output = PCA(2).fit_transform(pca_input)
# on obtient un tableau numpy avec deux colonnes car seulement les deux premières composantes principales ont été demanées

### GaussianMixture
type(pca_output), pca_output.shape
# on voit effectivement que cet ACP semble bien séparer deux clusters
plt.scatter(pca_output[:, 0], pca_output[:, 1], marker='.', color='green');
# pour les trouver ces deux clusters, Jake utilise une GaussianMixture
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(2)
# c'est ici que tout se passe
labels = gmm.fit(pca_input).predict(pca_input)
# la sortie est une association jour -> type
labels.shape, labels
