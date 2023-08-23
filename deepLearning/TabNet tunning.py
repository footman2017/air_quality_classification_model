import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pytorch_tabnet.augmentations import ClassificationSMOTE
import random

# Step 1: Preprocessing
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

# Step 2: TabNet Model Preparation
# Assuming X is your features and y is your target
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
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
)

# Step 3: K-Cross Validation
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
aug = ClassificationSMOTE(p=0.2)

max_epochs = 75
patience = 100
batch_size = 75 # 0.9967

# Hyperparameter tuning using random search
param_ranges = {
    'n_d': [4, 16],
    'n_a': [4, 16],
    'n_steps': [3, 7],
    'gamma': [1.0, 1.6]
}

num_trials = 20
best_accuracy = 0
best_params = {}

for _ in range(num_trials):
    param_dict = {
        'n_d': random.randint(*param_ranges['n_d']),
        'n_a': random.randint(*param_ranges['n_a']),
        'n_steps': random.randint(*param_ranges['n_steps']),
        'gamma': random.uniform(*param_ranges['gamma'])
    }
    clf = TabNetClassifier(
        **param_dict, cat_idxs=[], cat_dims=[], cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
    )

    fold_accuracies = []

    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=max_epochs, patience=patience,
            augmentations=aug, batch_size=batch_size
        )

        # Evaluate on validation set
        predictions = clf.predict(X_val)
        accuracy = np.mean(predictions == y_val)
        fold_accuracies.append(accuracy)

    average_accuracy = np.mean(fold_accuracies)
    
    # Update best_params and best_accuracy if current accuracy is better
    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_params = param_dict

    print(f"Average Validation Accuracy for Params {param_dict}: {average_accuracy:.4f}")

print("Best Hyperparameters:", best_params)
print("Best Validation Accuracy:", best_accuracy)