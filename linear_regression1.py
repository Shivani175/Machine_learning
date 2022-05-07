import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes() #dataset gets imported(this is a built-in dataset of sklearn)

#print(diabetes.keys())  # to see keys 
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']) 

print(diabetes.data) # to see data (several numpy arrays)
#print(diabetes.DESCR) 

diabetes_X = diabetes.data[:, np.newaxis, 2] #it takes just one feature of 2nd index from the data
#print(diabetes_X)
#diabetes_X = diabetes.data

diabetes_X_train = diabetes_X[:-30] # taking last 30 from diabetes_X data for training
diabetes_X_test = diabetes_X[-30:]# taking first 30 from diabetes_X data for testing

diabetes_Y_train= diabetes.target[:-30]
diabetes_Y_test= diabetes.target[-30:]
# x axis have features and y axis has labels. A linear line will be fitted with it. And then any value can be predicted.
model = linear_model.LinearRegression() # create linear model

model.fit(diabetes_X_train ,diabetes_Y_train)#fit the data(it means to make a line with data and that line will get saved in this linear model)
#give the data in model.fit to train the data. 
diabetes_Y_predicted = model.predict(diabetes_X_test)# it tells the line that is fit when we add feature to it what value it will predict.
print(diabetes_Y_predicted)
print("Mean squared error is: ", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted )) #calculating mean square error by giving actual values and predicted values
print("Weights : ", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test) # lets plot a scatter plot 
#plt.show()
plt.plot(diabetes_X_test,diabetes_Y_predicted)#it will show the best fit line
plt.show()

'''so basically we took just one feature from data{ line 14} so were able to 
make plot and see the best fit line but in order to make model more accurate we can take all the data {line 16} but then remove all plot lines from code. So by taking one feature 
mse =3035  and by taking complete dataset mse= 1826.53  '''


