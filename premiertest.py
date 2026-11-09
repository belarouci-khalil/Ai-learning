import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importer les donner 
data=pd.read_csv('loyesmaison.csv', sep=';')
#print(data)
# afficher nuage de points 
plt.scatter(data['surface'], data['prix'])

x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

#print(y)
#diviser le dataset
x_train , x_test, y_train , y_test=train_test_split(x,y,test_size=1.0/3)
#print(len(x))
#print(len(x_test))

# contruire le model 
regressor=LinearRegression()
regressor.fit(x_train, y_train)

#faire les prediction 
y_pred= regressor.predict(x_test)
print (regressor.predict([[300]]))