import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# Data constants (must match collect_data.py)
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30

def main():
    print("Loading data from disk...")
    label_map = {label:num for num, label in enumerate(actions)}
    
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
                # Check if file exists in case data collection crashed
                if os.path.exists(path):
                    res = np.load(path)
                    window.append(res)
            
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
            
    if len(sequences) == 0:
        print("Error: No data found! Please run collect_data.py first.")
        return

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    
    # Build the Deep Learning Model (LSTM for sequence data)
    print("Building LSTM Neural Network Model...")
    model = Sequential()
    # 126 is the number of coordinates we extract per frame (Left & Right Hands)
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Train the model
    print("Training model... (This will take a minute or two)")
    model.fit(X_train, y_train, epochs=200, verbose=1)
    
    # Save the model
    model.summary()
    model.save('isl_model.h5')
    print("Model successfully trained and saved as 'isl_model.h5'!")

if __name__ == '__main__':
    main()
