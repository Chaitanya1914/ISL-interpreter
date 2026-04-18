import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def normalize_batch(data):
    """Deep Normalization Engine (Rebuilds structural dependencies)"""
    reshaped = data.reshape(-1, 21, 3)
    mask = np.any(data != 0, axis=1)
    
    wrist = reshaped[:, 0, :][:, np.newaxis, :]
    shifted = reshaped - wrist
    dist = np.linalg.norm(shifted[:, 9, :], axis=1)[:, np.newaxis, np.newaxis]
    dist[dist < 1e-6] = 1e-6
    normalized = shifted / dist
    
    normalized[~mask] = 0
    return normalized.reshape(-1, 63)

def main():
    print("Connecting to Offline Backup Dataset Array...")
    class_col = ['class_name']
    lh_cols = [f'lh_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
    rh_cols = [f'rh_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]

    # Load from the renamed backup file we dynamically maintained
    df = pd.read_csv('landmarks_dataset.backup_data', usecols=class_col + lh_cols + rh_cols)
    
    # ----------------------------------------------------
    # FATAL FILTER: LIMITING AI TO EXACTLY 1 HUMAN POSE
    # ----------------------------------------------------
    # To prevent severe mathematical collapse (Classification requires a minimum of 2 states to exist),
    # I am configuring the model as a Binary "Hello" vs "Not Hello" system.
    
    hello_df = df[df['class_name'] == '48. Hello'].copy()
    hello_df['class_name'] = 'Hello'
    
    # Pull 1500 random background noise shapes to teach the AI what "Not Hello" looks like
    unknown_df = df[df['class_name'] != '48. Hello'].sample(n=1500, random_state=42).copy()
    unknown_df['class_name'] = '...' # Silent blank slate
    
    df = pd.concat([hello_df, unknown_df])

    classes = df['class_name'].values
    lh_data = df[lh_cols].values
    rh_data = df[rh_cols].values

    print("Synthesizing Base Multi-Hand Arrays...")
    lh_norm = normalize_batch(lh_data)
    rh_norm = normalize_batch(rh_data)

    X = np.hstack([lh_norm, rh_norm])
    y = classes

    valid_mask = np.any(X != 0, axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"Total isolated samples extracted: {len(X)}")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

    print("Executing Binary Classification (Hello vs Nothing)...")
    # Reduced layer size drastically because the brain is extremely tiny now
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200, 
        alpha=0.01,                           
        early_stopping=True,                   
        verbose=True, 
        random_state=42
    )
    
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"\n===========================================")
    print(f"Hello-Constraint Accuracy:   {test_acc * 100:.2f}%")
    print(f"===========================================\n")

    with open('isl_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'encoder': encoder}, f)
    print("Single-Word Initialization Complete!")

if __name__ == '__main__':
    main()
