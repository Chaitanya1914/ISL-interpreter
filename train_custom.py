import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

CSV_FILE = "my_isl_dataset.csv"

def normalize_batch(data):
    """Wrist-subtraction + Landmark-9 division for scale-invariant vectors"""
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
    print("Loading your custom webcam dataset...")
    df = pd.read_csv(CSV_FILE)

    lh_cols = [f'lh_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
    rh_cols = [f'rh_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]

    # Check which words were collected
    classes_found = df['class_name'].value_counts()
    print("\nDataset Summary:")
    print(classes_found)

    classes = df['class_name'].values
    lh_norm = normalize_batch(df[lh_cols].values)
    rh_norm = normalize_batch(df[rh_cols].values)

    X = np.hstack([lh_norm, rh_norm])
    y = classes

    valid_mask = np.any(X != 0, axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"\nTotal valid samples: {len(X)}")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

    print(f"Training on {len(X_train)} samples...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=400,
        alpha=0.005,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=True,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"\n===========================================")
    print(f"Training Accuracy:           {train_acc * 100:.2f}%")
    print(f"Unseen Test Accuracy:        {test_acc * 100:.2f}%")
    print(f"Classes Trained:             {list(encoder.classes_)}")
    print(f"===========================================\n")

    with open('isl_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'encoder': encoder}, f)
    print("Saved → isl_model.pkl  (Ready to use in gui_app.py!)")

if __name__ == '__main__':
    main()
