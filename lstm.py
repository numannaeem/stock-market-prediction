# Importing the Libraries
import warnings
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from turtle import color
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model

plt.style.use('seaborn-darkgrid')
plt.show(block=True)

warnings.filterwarnings("ignore")


def lstm():

    # Reading and preprocessing data
    df = pd.read_csv("Training Data/MSFT.csv", na_values=[
        'null'], index_col='Date', parse_dates=True, infer_datetime_format=True)

    print("Dataframe Shape: ", df. shape)
    print("Null Value Present: ", df.isnull().values.any())

    output_var = pd.DataFrame(df['Adj Close'])

    features = ['Open', 'High', 'Low', 'Volume']

    # Scaling the data to be used in the LSTM model.
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(df[features])
    feature_transform = pd.DataFrame(
        columns=features, data=feature_transform, index=df.index)
    feature_transform.head()

    # Splitting the data into 10 parts and using the first 9 parts for training and last part for testing.
    timesplit = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(
            train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(
            train_index): (len(train_index)+len(test_index))].values.ravel()

    # Reshaping the data to be used in the LSTM model.
    trainX = np.array(X_train)
    testX = np.array(X_test)
    X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

    # LSTM model defenition
    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(
        1, trainX.shape[1]), activation='relu', return_sequences=False))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error',
                 optimizer='adam', metrics=['accuracy'])
    plot_model(lstm, show_shapes=True, show_layer_names=True,
               to_file="LSTM_Model.png")

    history = lstm.fit(X_train, y_train, epochs=1,
                       batch_size=8, verbose=1, shuffle=False)
    lstm.save('LSTM_Model.h5')

    # LSTM Prediction
    y_pred = lstm.predict(X_test)
    mape = np.mean(np.abs((y_test-y_pred) / y_test)) * 100
    lstm_res = {
        "y_pred": y_pred,
        "y_test": y_test,
        "mape": mape
    }
    return lstm_res


def svm():
    df = pd.read_csv('./Training Data/MSFT.csv', index_col='Date',
                     parse_dates=True, infer_datetime_format=True)

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

    svm_res = {
        "actual_returns": df["Actual_Returns"],
        "predicted_returns": df["Predicted_Returns"],
        "accuracy": accuracy,
    }

    return svm_res


lstm_res = lstm()
print("LSTM Mean Absolute Percentage Error: " +
      str(round(lstm_res["mape"], 2)) + "%")
plt.plot(lstm_res["y_test"], label='True Value')
plt.plot(lstm_res["y_pred"], label='LSTM Value')
plt.title("Prediction by LSTM")
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()


# svm_res = svm()
# print("\nSVM Percentage Error: " +
#       str(round((100 - (svm_res["accuracy"])*100), 2)) + "%")
# figure(num=None, figsize=(40, 20), dpi=160, facecolor='w', edgecolor='k')
# svm_res["actual_returns"].plot(color='red')
# svm_res['predicted_returns'].plot(color='blue')
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Returns')
# plt.show()
