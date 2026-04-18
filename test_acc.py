import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

def normalize_batch(data):
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
    try:
        with open('isl_model.pkl', 'rb') as f:
            d = pickle.load(f)
        model = d['model']
        encoder = d['encoder']

        class_col = ['class_name']
        lh_cols = [f'lh_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
        rh_cols = [f'rh_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]

        df = pd.read_csv('landmarks_dataset.csv', usecols=class_col + lh_cols + rh_cols)
        target_words = ['48. Hello', '87. Doctor', '98. sick', '30. Hospital', '3. Medicine', '55. Thank you']
        df = df[df['class_name'].isin(target_words)]

        lh_norm = normalize_batch(df[lh_cols].values)
        rh_norm = normalize_batch(df[rh_cols].values)
        X = np.hstack([lh_norm, rh_norm])
        y = encoder.transform(df['class_name'].values)

        valid_mask = np.any(X != 0, axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        train_acc = model.score(X, y)
        print(f"OVERALL_ACCURACY:{train_acc * 100:.2f}%")
        
    except Exception as e:
        print("ERROR:", e)

if __name__ == '__main__':
    main()
