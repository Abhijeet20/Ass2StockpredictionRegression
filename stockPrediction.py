# Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
from sklearn import metrics
from IPython.display import display


start = datetime.datetime(2002, 8, 1)
end = datetime.datetime(2019, 9, 5)

df = web.DataReader("AAPL", 'yahoo', start, end)
display(df.tail())


window_size = 32 #Take window size for training data 
num_samples = len(df) - window_size
indices = np.arange(num_samples).astype(np.int)[:,None] + np.arange(window_size + 1).astype(np.int)

data = df['Adj Close'].values[indices] # Create the 2D matrix of training samples


# used to plot data
def plotdata(modelType,dataType, dataSet):
	df_model = df.copy()
	df_model.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
	
	
	if  dataType=="training":
		df_model = df_model.iloc[window_size:ind_split] # Past 32 days we don't know yet
		df_model['Adj Close Train'] = dataSet[:-window_size]
		df_model.plot(label='AAPL', figsize=(16,8), title=modelType+'|Training data', grid=True)
	else:
		df_model = df_model.iloc[ind_split+window_size:] # Past 32 days we don't know yet
		df_model['Adj Close Test'] = dataSet
		df_model.plot(label='AAPL', figsize=(16,8), title=modelType+'|Testing data', grid=True)
	plt.xlabel('Date')
	plt.ylabel('Adj Close')
	plt.show()



#Assign independent and dependent variables
X = data[:,:-1] # Each row represents 32 days in the past
y = data[:,-1] # Each output value represents the 33rd day


# Train and test split
split_fraction = 0.8
ind_split = int(split_fraction * num_samples)
X_train = X[:ind_split]
y_train = y[:ind_split]
X_test = X[ind_split:]
y_test = y[ind_split:]


#Model 1: Ridge
# Train
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# Infer
y_pred_train_ridge = ridge_model.predict(X_train)
y_pred_ridge = ridge_model.predict(X_test)


# Printing mean error
print('Mean Absolute Error for Ridge:', metrics.mean_absolute_error(y_test, y_pred_ridge))  
print('Mean Squared Error for Ridge:', metrics.mean_squared_error(y_test, y_pred_ridge))  
print('Root Mean Squared Error for Ridge:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge)))

#Plotting for training and test
plotdata("Ridge","training",y_pred_train_ridge)
plotdata("Ridge","test",y_pred_ridge)


# Model #2 - Lasso
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)

# Infer
y_pred_train_lasso = lasso_model.predict(X_train)
y_pred_lasso= lasso_model.predict(X_test)

# Printing mean error
print('Mean Absolute Error for Lasso:', metrics.mean_absolute_error(y_test, y_pred_lasso))  
print('Mean Squared Error for Lasso:', metrics.mean_squared_error(y_test, y_pred_lasso))  
print('Root Mean Squared Error for Lasso:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso)))

#Plotting for training and test
plotdata("Lasso","training",y_pred_train_lasso)
plotdata("Lasso","test",y_pred_lasso)


#Model 3-LinearRegression
linearRegression_model = LinearRegression()
linearRegression_model.fit(X_train, y_train)

# Infer
y_pred_train_linearRegression= linearRegression_model.predict(X_train)
y_pred_linearRegression= linearRegression_model.predict(X_test)

# Printing mean error
print('Mean Absolute Error for LinearRegression:', metrics.mean_absolute_error(y_test, y_pred_linearRegression))  
print('Mean Squared Error for LinearRegression:', metrics.mean_squared_error(y_test, y_pred_linearRegression))  
print('Root Mean Squared Error for LinearRegression:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_linearRegression)))

#Plotting for training and test
plotdata("LinearRegression","training",y_pred_train_linearRegression)
plotdata("LinearRegression","test",y_pred_linearRegression)



