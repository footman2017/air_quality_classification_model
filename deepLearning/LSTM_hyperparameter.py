import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from kerastuner import HyperModel
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from kerastuner.tuners import BayesianOptimization

df = pd.read_csv('combined_data ver 1.1.csv')
df = df.replace('---', pd.NA)
df = df.dropna()
df.reset_index(drop=True, inplace=True)

df['pm10'] = df['pm10'].astype(int)
df['pm25'] = df['pm25'].astype(int)
df['so2'] = df['so2'].astype(int)
df['co'] = df['co'].astype(int)
df['o3'] = df['o3'].astype(int)
df['no2'] = df['no2'].astype(int)

scaler = MinMaxScaler()
X = scaler.fit_transform(df.drop('categori', axis=1).values)

encoder = LabelEncoder()
y = encoder.fit_transform(df['categori'].values)
y = to_categorical(y)

# Define the Hypermodel
class CategoriClassificationHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()

        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32),
                       input_shape=self.input_shape,
                       return_sequences=True))

        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32)))

        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


# Bayesian Optimization
input_shape = (X.shape[1], 1)
hypermodel = CategoriClassificationHyperModel(input_shape=input_shape)

tuner = BayesianOptimization(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    seed=42,
    directory='output_bayesian',
    project_name='CategoriClassification'
)

# Assuming a split between train and test data, here we use 80% of data for training
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(
    f"The optimal number of units in the LSTM layers is {best_hps.get('units')} and the optimal optimizer is {best_hps.get('optimizer')}.")