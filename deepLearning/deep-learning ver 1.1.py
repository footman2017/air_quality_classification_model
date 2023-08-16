from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

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

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(df.drop('categori', axis=1).values)

# Reshape data for CNN
X_cnn = X.reshape(-1, 6, 1)

# Convert categories to integers
encoder = LabelEncoder()
y = encoder.fit_transform(df['categori'].values)
y = to_categorical(y)


def create_lstm_model():
    model = Sequential()

    # sebelum hyperparameter tunning
    # model.add(LSTM(50, input_shape=(X.shape[1], 1), return_sequences=True))
    # model.add(LSTM(50))

    # sesudah hyperparameter tunning
    model.add(LSTM(128, input_shape=(X.shape[1], 1), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(3, activation='softmax'))  # 3 categories
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


# Define the CNN model
def create_cnn_model(filters=1, kernel_size=2, dense_units=1, optimizer='sgd'):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size,
              activation='relu', input_shape=(6, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

acc_per_fold = []

for fold, (train, test) in enumerate(kfold.split(X, np.argmax(y, axis=1))):
    print(f"\n---- Training on Fold {fold + 1} ----")

    model = create_lstm_model()
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    # LSTM expects input in the shape (samples, timesteps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print("Training the model...")
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # model.fit(X_train, y_train, epochs=1000, batch_size=10, verbose=1, callbacks=[early_stopping])
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
    # model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)
    # model.fit(X_train, y_train, epochs=20, batch_size=10, verbose=1)

    print("Evaluating the model...")
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(
        f"Score for fold {fold + 1}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%")
    acc_per_fold.append(scores[1] * 100)

lstm_acc = f"\nLSTM Average accuracy over {n_splits}-fold CV: {np.mean(acc_per_fold):.2f}%"
# print(
#     f"\nLSTM Average accuracy over {n_splits}-fold CV: {np.mean(acc_per_fold):.2f}%")


# for cnn
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

acc_per_fold = []

for fold, (train, test) in enumerate(kfold.split(X_cnn, np.argmax(y, axis=1))):
    print(f"\n---- Training on Fold {fold + 1} ----")

    # model = create_cnn_model(112, 2, 88, 'adam')
    model = create_cnn_model()
    X_train, X_test = X_cnn[train], X_cnn[test]
    y_train, y_test = y[train], y[test]

    print("Training the model...")
    # model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

    print("Evaluating the model...")
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(
        f"Score for fold {fold + 1}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%")
    acc_per_fold.append(scores[1] * 100)

print(
    f"\nCNN Average accuracy over {n_splits}-fold CV: {np.mean(acc_per_fold):.2f}%")

print(lstm_acc)
