# Importing the Libraries
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from keras.layers import LSTM, Dense
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
warnings.filterwarnings("ignore")


df = pd.read_csv("Training Data/MSFT.csv", na_values=[
    'null'], index_col='Date', parse_dates=True, infer_datetime_format=True)

# Print the shape of Dataframe  and Check for Null Values
print("Dataframe Shape: ", df. shape)
print("Null Value Present: ", df.isnull().values.any())

output_var = pd.DataFrame(df['Adj Close'])
# Selecting the Features
features = ['Open', 'High', 'Low', 'Volume']

scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform = pd.DataFrame(
    columns=features, data=feature_transform, index=df.index)
feature_transform.head()

timesplit = TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(
        train_index): (len(train_index)+len(test_index))]
    y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(
        train_index): (len(train_index)+len(test_index))].values.ravel()

trainX = np.array(X_train)
testX = np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

lstm = Sequential()
lstm.add(LSTM(32, input_shape=(
    1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
# plot_model(lstm, show_shapes=True, show_layer_names=True,
#             to_file="LSTM_Model.png")

history = lstm.fit(X_train, y_train, epochs=5,
                    batch_size=8, verbose=1, shuffle=False)

# LSTM Prediction
y_pred = lstm.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred)
print("\n\nLSTM Model Accuracy: " + str(100-round(mape*100, 2)) + "%")

# Predicted vs True Adj Close Value – LSTM
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.title("Prediction by LSTM")
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()
    