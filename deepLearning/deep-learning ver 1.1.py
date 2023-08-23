from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from pytorch_tabnet.augmentations import ClassificationSMOTE

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
    model.add(LSTM(50, input_shape=(X.shape[1], 1), return_sequences=True))
    model.add(LSTM(50))

    # sesudah hyperparameter tunning
    # model.add(LSTM(128, input_shape=(X.shape[1], 1), return_sequences=True))
    # model.add(LSTM(128))
    model.add(Dense(3, activation='softmax'))  # 3 categories
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


# Define the CNN model
def create_cnn_model(filters=32, kernel_size=2, dense_units=50, optimizer='adam'):
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


#tabnet variable
n_d = 8  # Width of the decision prediction layer. Bigger values gives more capacity to the model with the risk of overfitting.
n_a = 8  # Width of the attention embedding for each mask. According to the paper n_d=n_a is usually a good choice.
n_steps = 3  # Number of steps in the architecture (usually between 3 and 10)
gamma = 1.3  # This is the coefficient for feature reusage in the masks. A value close to 1 will make mask selection least correlated between layers. Values range from 1.0 to 2.0.

clf = TabNetClassifier(
    n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
    cat_idxs=[], cat_dims=[], cat_emb_dim=1,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15, verbose=0
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
aug = ClassificationSMOTE(p=0.2)

max_epochs = 75
patience = 100
batch_size = 75 # 0.9967

acc_per_fold_lstm = []
acc_per_fold_cnn = []
acc_per_fold_tabnet = []

for fold, (train, test) in enumerate(kfold.split(X, np.argmax(y, axis=1))):
    print(f"\n---- Training on Fold {fold + 1} ----")

    model_lstm = create_lstm_model()
    model_cnn = create_cnn_model()

    #cnn sesudah hyperparameter tunning
    # model_cnn = create_cnn_model(112, 2, 88, 'adam')

    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    y_train_tabnet = np.argmax(y_train, axis=1)
    y_test_tabnet = np.argmax(y_test, axis=1)

    clf.fit(
        X_train, y_train_tabnet,
        eval_set=[(X_test, y_test_tabnet)],
        max_epochs=max_epochs, patience=patience,
        augmentations=aug, batch_size=batch_size
    )

    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == y_test_tabnet)
    acc_per_fold_tabnet.append(accuracy * 100)

    # LSTM expects input in the shape (samples, timesteps, features)
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print("Training the model...")

    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # model_lstm.fit(X_train, y_train, epochs=1000, batch_size=10, verbose=1, callbacks=[early_stopping])
    # model_lstm.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)
    # model_lstm.fit(X_train, y_train, epochs=20, batch_size=10, verbose=1)

    model_lstm.fit(X_train_lstm, y_train, epochs=100, batch_size=10, verbose=0)

    X_train = X_train.reshape(-1, 6, 1)
    X_test = X_test.reshape(-1, 6, 1)

    model_cnn.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)


    print("Evaluating the model...")
    scores = model_lstm.evaluate(X_test_lstm, y_test, verbose=1)
    print(
        f"Score for fold {fold + 1}: {model_lstm.metrics_names[0]} of {scores[0]}; {model_lstm.metrics_names[1]} of {scores[1]*100}%")
    acc_per_fold_lstm.append(scores[1] * 100)

    scores = model_cnn.evaluate(X_test, y_test, verbose=0)
    print(
        f"Score for fold {fold + 1}: {model_cnn.metrics_names[0]} of {scores[0]}; {model_cnn.metrics_names[1]} of {scores[1]*100}%")
    acc_per_fold_cnn.append(scores[1] * 100)


# lstm_acc = f"\nLSTM Average accuracy over {n_splits}-fold CV: {np.mean(acc_per_fold):.2f}%"
print(acc_per_fold_lstm)
print(
    f"\nLSTM Average accuracy over {n_splits}-fold CV: {np.mean(acc_per_fold_lstm):.2f}%")

print(acc_per_fold_cnn)
print(
    f"CNN Average accuracy over {n_splits}-fold CV: {np.mean(acc_per_fold_cnn):.2f}%")


print(acc_per_fold_tabnet)
print(
    f"Tabnet Average accuracy over {n_splits}-fold CV: {np.mean(acc_per_fold_tabnet):.2f}%")


# for cnn
# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# acc_per_fold = []

# for fold, (train, test) in enumerate(kfold.split(X_cnn, np.argmax(y, axis=1))):
#     # print(f"\n---- Training on Fold {fold + 1} ----")

#     model = create_cnn_model(112, 2, 88, 'adam')
#     # model = create_cnn_model(96, 5, 88, 'nadam')        #97.82
#     # model = create_cnn_model(128, 1, 120, 'nadam')    
#     # model = create_cnn_model(48, 3, 72, 'rmsprop')     #highest 96.31
#     # model = create_cnn_model(80, 2, 56, 'rmsprop')   #hignest 97.10
    
#     # model = create_cnn_model()
#     X_train, X_test = X_cnn[train], X_cnn[test]
#     y_train, y_test = y[train], y[test]

#     # print("Training the model...")
#     # model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)
#     model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

#     # print("Evaluating the model...")
#     scores = model.evaluate(X_test, y_test, verbose=0)
#     print(
#         f"Score for fold {fold + 1}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%")
#     acc_per_fold.append(scores[1] * 100)

# print(
#     f"\nCNN Average accuracy over {n_splits}-fold CV: {np.mean(acc_per_fold):.2f}%")

# print(lstm_acc)
