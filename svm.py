# Machine learning
import datetime
from turtle import color
import warnings
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use('seaborn-darkgrid')
plt.show(block=True)

# To ignore warnings
warnings.filterwarnings("ignore")

# Read the csv file using read_csv
df = pd.read_csv('Training Data/MSFT.csv')

# Changes The Date column as index columns
df.index = pd.to_datetime(df['Date'])

# drop The original date column
df = df.drop(['Date'], axis='columns')

# Create predictor variables
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

# Store all predictor variables in a variable X
X = df[['Open-Close', 'High-Low']]
X.head()

# Defining target  variables
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

split_percentage = 0.8
split = int(split_percentage*len(df))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]

# Support vector classifier
print("\nTraining the SVM model...")
cls = SVC().fit(X_train, y_train)
print("\nTesting the SVM model...")
accuracy = cls.score(X_test, y_test)
print("\nPlotting test results...")

# Calculate Actual returns
df['Return'] = df.Close.pct_change()
df['Actual_Returns'] = df['Return'].cumsum()

# Calculate Predicted returns
df['Predicted_Signal'] = cls.predict(X)
df['Single_Pred_Returns'] = df.Return * df.Predicted_Signal.shift(1)
df['Predicted_Returns'] = df['Single_Pred_Returns'].cumsum()

print('\nFinished')
print("\nTest Accuracy: " + str(round((accuracy*100), 2)) + "%")
figure(num=None, figsize=(40, 20), dpi=160, facecolor='w', edgecolor='k')
df['Actual_Returns'].plot(color='red')
df['Predicted_Returns'].plot(color='blue')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Returns')
plt.show()
