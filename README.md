# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

## Neural Network Model
Here we have taken the price of google stock dataset 
![image](https://user-images.githubusercontent.com/94219582/236667144-d6ab6ab6-459e-4c60-90a4-4ba63ce5c8aa.png)


## DESIGN STEPS

### STEP 1:
Import the necessary tensorflow modules

### STEP 2:
Load the stock dataset

### STEP 3:
Fit the model and then predict

### STEP 4:
Create a model with the desired number of neurons and one output neuron.
### STEP 5:
Follow the same steps to create the Test data. But make sure you combine the training data with the test data.

### STEP 6:
Make Predictions and plot the graph with the Actual and Predicted values.

### 
## PROGRAM
```
Developed By: Sithi Hajara I
Register No.: 212221230102
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from tensorflow.keras import models
from keras.models import Sequential
from tensorflow.keras import layers

df = pd.read_csv("/content/trainset.csv")

df.columns

df.head()

train_set = df.iloc[:,1:2].values

type(train_set)

train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape

length = 60
n_features = 1

model = Sequential()
model.add(layers.SimpleRNN(50,input_shape=(60,1)))

model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mse',metrics ='accuracy')

model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

import sklearn.metrics as metrics

metrics[['loss','val_loss']].plot()

dataset_test = pd.read_csv("/content/testset.csv")

test_set = dataset_test.iloc[:,1:2].values

test_set.shape

dataset_total = pd.concat((df['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

```
## OUTPUT

### True Stock Price, Predicted Stock Price vs time
![graph](https://user-images.githubusercontent.com/94219582/236667059-e52a1e7f-14ba-42b4-9773-119d2db29ab2.png)


### Mean Square Error
![epoc](https://user-images.githubusercontent.com/94219582/236667065-a69f9d28-e4ba-4e96-89b8-c8b4d765a330.png)


### RESULT
A Recurrent Neural Network model for stock price prediction is developed.

