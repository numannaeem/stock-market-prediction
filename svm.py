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


# To ignore warnings
warnings.filterwarnings("ignore")

# Read the csv file using read_csv
df = pd.read_csv('./Training Data/MSFT.csv', index_col='Date',
                 parse_dates=True, infer_datetime_format=True)

# Changes The Date column as index columns

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
accuracy = cls.score(X_test, y_test)

# Calculate Actual returns
df['Return'] = df.Close.pct_change()
df['Actual_Returns'] = df['Return'].cumsum()

# Calculate Predicted returns
df['Predicted_Signal'] = cls.predict(X)
df['Single_Pred_Returns'] = df.Return * df.Predicted_Signal.shift(1)
df['Predicted_Returns'] = df['Single_Pred_Returns'].cumsum()


print("\nSVM Model Accuracy: " + str(round((accuracy*100), 2)) + "%")
plt.plot(df['Actual_Returns'], label='True Returns')
plt.plot(df['Predicted_Returns'], label='SVM Returns')
plt.title("Prediction by SVM")
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()
