import numpy as np
import n5
from n5 import nn5 , f5
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Correct 2D Arrays format for inputs and outputs
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    print("--- Phase 1: Creating and Training the Original Model ---")
    model = nn5.Sequential()
    model.add(nn5.Dense(input_size=2, output_size=4, activation='tanh'))
    model.add(nn5.Dense(input_size=4, output_size=1, activation='sigmoid'))

    model.summary()

    # Train and save in your brand new custom .n5 format
    model.train(X_train, y_train, epochs=300, lr_init=0.5, loss_type='mse')
    model.save("module.n5")

    print("\n--- Phase 2: Loading the .n5 file directly inside a fresh execution ---")
    # Load it directly without defining architecture
    module = nn5.Sequential.load("module.n5")

    # The loaded model automatically remembers everything
    module.summary()

    # Make predictions instantly
    live_predictions = module.predict(X_train)
    print("Direct predictions from the loaded .n5 file:")
    print(live_predictions)
