


# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing dataset
dataset=pd.read_csv(r"C:\Users\Hemant Ojha\OneDrive\Desktop\Finlatics1\MLResearch\MLResearch\advertising_sales_data.csv")


# Q 1. What is the average amount spent on TV advertising in the dataset?

average_amount_spent_on_tv=dataset['TV'].mean()
print(average_amount_spent_on_tv)

# Ans :- Average amount spent on TV advertising in the dataset is 147.0425



# Q2. What is the correlation between radio advertising expenditure and product sales?

print(dataset.info())

print(dataset.isnull().sum())  # so 2 missing values in Radio column

# impute the missing value in Radio column

dataset['Radio']=dataset['Radio'].fillna(dataset['Radio'].mean())

# creating variable
numeric_data=dataset[['Radio','Sales']].select_dtypes(include=['float64'])
# Creating correlation matrix of radio advertising expenditure and product sales
corr_matrix=numeric_data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='PuBuGn',fmt='.2f')
plt.title('correlation between radio advertising expenditure and product sales')
plt.show()

# Ans:- Correlation between radio advertising expenditure and product sales is 0.35



# Q3. Which advertising medium has the highest impact on sales based on the dataset?


# Correlation between TV ,Radio,Newspaper and sales

# creating variable
numeric_data=dataset[['TV','Radio','Newspaper','Sales']].select_dtypes(include=['float64'])
# Creating correlation matrix of advertising expenditures(TV,Radio,Newspaper) and product sales
corr_matrix=numeric_data.corr()
print(corr_matrix)
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='PuBuGn',fmt='.2f')
plt.title('correlation between advertising expenditures(TV,Radio,Newspaper) and product sales')
plt.show()

'''
output:- 
Correlation between TV advertising and sales is 0.90
Correlation between radio advertising expenditure and product sales is 0.35
Correlation between Newspaper advertising and sales is 0.16
'''

# Ans:- So correlation between TV advertising and sales is high. Therefore TV advertising has highest impact on product sales


'''
Q 4. Plot a linear regression line that includes all variables (TV, Radio, Newspaper) to predict 
Sales, and visualize the model's predictions against the actual sales values.
'''


# Splitting dataset into independent and dependent set

x=dataset.iloc[:,[1,2,3]].values
y=dataset.iloc[:,4].values

print(x)

print(y)

# Hiding the missing value in independent variable

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(x[:,1:4])
x[:,1:4]=imputer.transform(x[:,1:4])
print(x)

# split dataset into train and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

print(x_train)

print(x_test)

print(y_train)

print(y_test)

# Training the model

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

# Predicting the test set

y_pred=regressor.predict(x_test)

# Visualizing the training set result

for i in range(x.shape[1]):
    plt.figure()
    plt.scatter(x_train[:, i], y_train, color='red', label='Training Data')

    # Generate predicted values for the current feature
    x_pred = np.linspace(x_train[:, i].min(), x_train[:, i].max(), 100).reshape(-1, 1)
    regressor.fit(x_train[:, i].reshape(-1, 1), y_train)
    y_pred = regressor.predict(x_pred)

    plt.plot(x_pred, y_pred, color='blue', label='Regression Line')
    feature_names = ['TV', 'Radio', 'Newspaper']

    # Use the feature name from the list
    plt.title(f'{feature_names[i]} vs Sales (Training Set)')
    plt.xlabel(feature_names[i])
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

# Visualizing the testing set result

for i in range(x.shape[1]):
    plt.figure()
    plt.scatter(x_test[:, i], y_test, color='red', label='Testing Data')

    # Generate predicted values for the current feature
    x_pred = np.linspace(x_test[:, i].min(), x_test[:, i].max(), 100).reshape(-1, 1)
    regressor.fit(x_train[:, i].reshape(-1, 1), y_train)
    y_pred = regressor.predict(x_pred)

    plt.plot(x_pred, y_pred, color='blue', label='Regression Line')
    feature_names = ['TV', 'Radio', 'Newspaper']

    # Use the feature name from the list
    plt.title(f'{feature_names[i]} vs Sales (Testing Set)')
    plt.xlabel(feature_names[i])
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
    
    
    
'''
Q5.	How would sales be predicted for a new set of advertising expenditures: $200 on TV, $40 on 
Radio, and $50 on Newspaper?
'''

# Since we already trained the model in Q 4 . therefore we need only include the new advertising expenditures on TV,Radio and Newspaper

new_data = [[200, 40, 50]]  # TV, Radio, Newspaper
predicted_sales = regressor.predict(new_data)

# Display the prediction
print(f"Predicted sales: ${predicted_sales[0]:.2f}")

# Ans:- Predicted sales for new set is $19.73




# Q6. How does the performance of the linear regression model change when the dataset is normalized?


from sklearn.preprocessing import StandardScaler

# Normalizing the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Training the model on normalized data
regressor = LinearRegression()
regressor.fit(x_train_scaled, y_train)
print(x_train_scaled)
print(y_train)
print(x_test_scaled)
print(y_test)




'''
Q7.	What is the impact on the sales prediction when only radio and newspaper advertising expenditures
are used as predictors?
'''

# Here we will compare the R² Scores and MSE for including TV and excluding TV from dataset



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Hemant Ojha\OneDrive\Desktop\Finlatics1\MLResearch\MLResearch\advertising_sales_data.csv")

# Split the data into independent and dependent set
x=dataset.iloc[:,[1,2,3]].values
y=dataset.iloc[:,4].values

# Hinding the missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
x = imputer.fit_transform(x)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict values for the test set
y_pred = regressor.predict(x_test)

# Evaluate the model
train_score = regressor.score(x_train, y_train)
test_score = regressor.score(x_test, y_test)

print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")

# Calculate  MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# output
# Train Score: 0.9001598701331188
# Test Score: 0.9059117026092903
# Mean Squared Error: 2.9074318865000572


# Now we find MSE and R² Scores without TV advertising


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Hemant Ojha\OneDrive\Desktop\Finlatics1\MLResearch\MLResearch\advertising_sales_data.csv")

# Split the data into independent and dependent set
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

# Hinding the missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
x = imputer.fit_transform(x)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict values for the test set
y_pred = regressor.predict(x_test)

# Evaluate the model
train_score = regressor.score(x_train, y_train)
test_score = regressor.score(x_test, y_test)

print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")

# Calculate  MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# output
# Train Score: 0.11726533796819838
# Test Score: 0.10998637077378093
# Mean Squared Error: 27.502400158082317


# So from both we see that MSE between predicted y_pred and actual y_test is less with TV advertising 
# but it is high without TV so clearly we can say that more impact of TV advertising on product sales



























