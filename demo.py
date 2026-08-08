import numpy as np
from n5 import nn5, f5

if __name__ == '__main__':
    print('==================================================================')
    print('🚀 n5 lightweight demo: Dense, Dropout, Conv2D, Pooling, Custom Layer')
    print('==================================================================')
    
    # --- 1. Dense Example ---
    print('\n--- [1/5] Training Dense Model ---')
    X = np.random.randn(4, 3)
    y = np.random.randint(0, 2, (4,1))
    
    model = nn5.Sequential()
    model.add(nn5.Dense(3, 8, activation='gelu'))
    model.add(nn5.Dense(8, 4, activation='leaky_relu'))
    model.add(nn5.Dense(4, 1, activation='sigmoid'))
    
    model.summary()
    model.train(X, y, epochs=3, lr_init=0.01, loss_type='bce')
    
    # --- 2. Dropout Example ---
    print('\n--- [2/5] Testing Dropout Layer ---')
    X2 = np.random.randn(2,4)
    drop = nn5.Dropout(rate=0.5, seed=42)
    print('Dropout forward (training):\n', drop.forward(X2, training=True))
    print('Dropout backward sample:\n', drop.backward(np.ones_like(X2)))
    
    # --- 3. Conv2D + Pooling Example (toy) ---
    print('\n--- [3/5] Testing Conv2D + Pooling ---')
    Xin = np.random.randn(1,8,8,1) # batch=1, 8x8 grayscale
    conv = nn5.Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, activation='relu')
    out = conv.forward(Xin)
    print('Conv2D output shape:', out.shape)
    
    pool = nn5.MaxPool2D(pool_size=2, stride=2)
    p = pool.forward(out)
    print('After MaxPool2D shape:', p.shape)
    
    # --- 4. Custom Layer Demo ---
    print('\n--- [4/5] Testing Integrated SquareLayer ---')
    seq2 = nn5.Sequential()
    seq2.add(nn5.Dense(3,3, activation='relu'))
    seq2.add(nn5.SquareLayer()) # Appelé depuis nn5
    x3 = np.random.randn(2,3)
    print('Custom layer pipeline output:\n', seq2.predict(x3))
    
    # --- 5. Save/load state demo ---
    print('\n--- [5/5] Testing Model Serialization (Save/Load) ---')
    model.save_state('demo_model.npz')
    loaded = nn5.Sequential.load_state('demo_model.npz')
    print('Loaded sequential layers:', [l.__class__.__name__ for l in loaded.layers])
    print('\n🎉 Demo finished successfully!')
