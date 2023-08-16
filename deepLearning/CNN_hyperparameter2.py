import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from kerastuner.tuners import RandomSearch

# Load and preprocess data
df = pd.read_csv('combined_data ver 1.1.csv')
df = df.replace('---', pd.NA).dropna().reset_index(drop=True)

# Convert columns to integers
cols = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
for col in cols:
    df[col] = df[col].astype(int)

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(df.drop('categori', axis=1).values)
X_cnn = X.reshape(-1, 6, 1)

# Convert categories to integers
encoder = LabelEncoder()
y = encoder.fit_transform(df['categori'].values)
y = to_categorical(y)

# Define model with hyperparameters


def build_model(hp):
    model = Sequential()
    model.add(Conv1D(filters=hp.Int('filters', min_value=16, max_value=128, step=16),
                     kernel_size=hp.Int(
                         'kernel_size', min_value=1, max_value=5, step=1),
                     activation='relu',
                     input_shape=(6, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=hp.Int('dense_units', min_value=8, max_value=128, step=16),
                    activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'nadam', 'rmsprop']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Setup Keras Tuner and hyperparameter search
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_models = []
for fold, (train, test) in enumerate(kfold.split(X_cnn, np.argmax(y, axis=1))):
    print(f"\n---- Hyperparameter Tuning on Fold {fold + 1} ----")

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='output',
        project_name=f'AirQualityCNN_Fold{fold + 1}'
    )

    X_train, X_test = X_cnn[train], X_cnn[test]
    y_train, y_test = y[train], y[test]

    tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=0)

    # Display best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    Best hyperparameters for fold {fold + 1}:
    Filters: {best_hps.get('filters')}
    Kernel Size: {best_hps.get('kernel_size')}
    Dense Units: {best_hps.get('dense_units')}
    Optimizer: {best_hps.get('optimizer')}
    """)

    # You can continue to train the best model for more epochs or save it if you want.
    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test), verbose=0)
    best_models.append(best_model)
