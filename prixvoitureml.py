import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_excel('dataset_prix_voitures.xlsx')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

#gerer les variables categoriques
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [("marque et modele ", OneHotEncoder(handle_unknown='ignore'), [0, 1])],
    remainder='passthrough'
)
x=ct.fit_transform(x)



x_train, x_test, y_train , y_test= train_test_split(x,y, test_size=0.2)



regressor= LinearRegression()



regressor.fit(x_train, y_train)



y_pred=regressor.predict(x_test)



print(regressor.predict([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2019.0, 45000.0, 130.0]]))