# n5 Deep Learning Framework 🚀

A lightweight, modular deep-learning micro-framework written from scratch in Python + NumPy.
This README is written for newcomers and contains quick-start examples, descriptions of the main
components, and safe usage notes.

---

## نظرة سريعة بالعربية — Arabic quick overview

n5 هو إطار تعلّم عميق خفيف ومبسّط مكتوب بلغة Python و NumPy. يهدف إلى أن يكون تعليميًا، واضح البُنية، وسهل التوسيع لكتابة طبقات مُخصّصة دون الحاجة لتعديل المكتبة نفسها.

الميزات الرئيسية:
- محرك رياضي (`f5`) يحوي الدوال والتفاضلات (ReLU, Sigmoid, GELU, ...).
- محرك بنيوي (`nn5`) يحوي طبقات جاهزة: Dense, Conv2D, MaxPool2D, AvgPool2D, Dropout, Flatten.
- واجهة بسيطة لكتابة طبقات مخصّصة (Custom Layers) — أي كائن يوفّر `forward`, `backward`, و`update` يمكن إضافته لـ Sequential.
- حفظ واستعادة آمن للـ weights باستخدام `save_state` / `load_state` (.npz). خيار `save`/`load` بالـ pickle متاح لكنه غير آمن مع ملفات غير موثوقة.

---

## Quick Start (English)

Requirements:
- Python 3.8+
- NumPy

Install (example):

```bash
pip install numpy
```

Basic usage examples (Python):

1) Dense model (architecture + train)

```python
import numpy as np
from n5 import nn5

# Build model
model = nn5.Sequential()
model.add(nn5.Dense(3, 8, activation='gelu'))
model.add(nn5.Dense(8, 4, activation='leaky_relu'))
model.add(nn5.Dense(4, 1, activation='sigmoid'))

# Inspect
model.summary()

# Create synthetic data
X_train = np.random.randn(100, 3)
y_train = np.random.randint(0, 2, (100, 1))

# Train
model.train(X_train, y_train, epochs=50, lr_init=0.01, loss_type='bce')

# Predict
preds = model.predict(X_train)
```

2) Convolution + Pool example (toy)

```python
import numpy as np
from n5 import nn5

# Input: batch x H x W x C
X = np.random.randn(2, 8, 8, 1)
conv = nn5.Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, activation='relu')
pooled = nn5.MaxPool2D(pool_size=2, stride=2)

Z = conv.forward(X)
P = pooled.forward(Z)
print('Conv output', Z.shape)
print('Pooled output', P.shape)
```

3) Dropout usage

```python
from n5 import nn5
import numpy as np

x = np.random.randn(4, 8)
drop = nn5.Dropout(rate=0.5, seed=42)
# training forward
y_train = drop.forward(x, training=True)
# inference forward
y_infer = drop.forward(x, training=False)
```

4) Custom layer example (SquareLayer)

```python
# Define a custom layer exactly like the example in n5.py
class SquareLayer(nn5.Layer):
    def __init__(self):
        self.X = None
    def forward(self, X):
        self.X = X
        return X**2
    def backward(self, da):
        return 2 * self.X * da

# Use it in a pipeline
model = nn5.Sequential()
model.add(nn5.Dense(3,3, activation='relu'))
model.add(SquareLayer())
out = model.predict(np.random.randn(2,3))
```

5) Save / Load state (recommended)

```python
# Save weights and simple metadata in a portable .npz
model.save_state('model_demo.npz')

# Reconstruct known layers from the .npz file
loaded = nn5.Sequential.load_state('model_demo.npz')
```

Notes:
- save/load (.n5) use pickle and will execute code during load. Only load pickle files you trust.
- `save_state`/`load_state` (.npz) is the safe and recommended mechanism for sharing model parameters.

---

## Repository structure

```text
├── check_point/     # Legacy snapshots (use with caution)
├── n5.py            # Core framework (f5 math + nn5 structural layers)
├── main.py          # Example entry (training scripts, demos)
├── model.n5         # Example pickle-packed model (if any)
├── LICENSE          # Proprietary license and usage clauses
└── README.md        # This file
```

---

## License & Usage (مختصر)

This project uses a restrictive proprietary license: non-commercial educational use is allowed; commercial use, redistribution, AI-training ingestion, or military/non-peaceful use are prohibited without explicit written permission from the copyright owner (soufian2024). See LICENSE for full details.

If you need a different license for collaboration or contributions, contact the repository owner.

---

## Contributing

If you want to contribute, open an issue describing the change and contact the owner to request permission for usage beyond the allowed scope. Adding tests for new layers and optimizing Conv2D implementations (vectorized/im2col) is welcome.

---

If you want, I can now open a Pull Request that updates README.md, the code (n5.py), and LICENSE together. Or I can open separate PRs. Tell me how you'd like to proceed.