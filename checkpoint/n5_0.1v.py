# ==============================================================================
# COPYRIGHT, TRADEMARK AND PROPERTY LICENSE NOTICE
# ==============================================================================
# Copyright (c) 2024-2026 soufian2024. All rights reserved.
#
# LICENSING TERMS:
# This source code is released exclusively for educational, personal research,
# and non-commercial development purposes.
#
# STRICT NON-COMMERCIAL RESTRICTIONS:
# 1. Any commercial deployment, corporate usage, or integration into paid
#    software products is STRICTLY PROHIBITED.
# 2. Reselling, sub-licensing, or distributing this framework for profit by
#    anyone other than the explicit copyright owner (soufian2024) is forbidden.
#
# Any authorized open-source fork or study copy must retain this original
# copyright header intact without modification.
# ==============================================================================


##################################################
##                                              ##
##                           ##########         ##
##          ##  ######       ##                 ##
##          ####     ##      ##                 ##
##          ##       ##      ##########         ##
##          ##       ##              ##         ##
##          ##       ##              ##         ##
##          ##       ##      ##      ##         ##
##          ##       ##      ##########         ##
##                                              ##
##################################################




import numpy as np

class f5:
    def relu(Z):
        result = np.where(Z > 0, Z, 0)
        return result

    def d_relu(Z):
        result = np.where(Z > 0, 1, 0)
        return result

    def leaky_relu(Z):
        result = np.where(Z > 0, Z, Z * 0.01)
        return result

    def d_leaky_relu(Z):
        result = np.where(Z > 0, 1, 0.01)
        return result

    def sigmoid(Z):
        result = 1 / (1 + np.exp(-Z))
        return result

    def loss_mse(y_predict,y_true):
        mse = np.mean((y_predict - y_true)**2)
        return mse

    def d_loss_mse(y_predict,y_true):
        d_mse = np.mean((y_predict - y_true)*2)
        return d_mse


class nn5:
    class Dense:
        def __init__(self, input_size, output_size, activation='relu'):
            self.W = 2 * (np.random.rand(input_size, output_size) - 0.5)
            self.B = np.zeros((1, output_size))
            self.activation = activation

        def forward(self, X):
            self.X = X
            self.Z = np.dot(X, self.W) + self.B

            if self.activation == 'relu':
                self.A = f5.relu(self.Z)
                return self.A
            elif self.activation == 'leaky_relu':
                self.A = f5.leaky_relu(self.Z)
                return self.A
            elif self.activation == 'sigmoid':
                self.A = f5.sigmoid(self.Z)
                return self.A

    class Sequential:
        def __init__(self):
            self.layers = []
            self.history = []

        def add(self, layer):
            self.layers.append(layer)

        def predict(self, X):
            for layer in self.layers:
                X = layer.forward(X)
            return X



        def train(self, X, y, epochs=100, lr_init=0.01):

            epochs = epochs + 1


            # Sou Studio
            # can you make caine from The Amazing Digital Circus with this ? I mean n5,
            # and what about cyn from morder drones ? can you make it ?

            m = X.shape[1]

            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            lr = lr_init

            num_layers = len(self.layers)

            weights = []
            biases = []

            for  i in range(num_layers):
                weights.append(self.layers[i].W)
                biases.append(self.layers[i].B)

            m_w = [np.zeros_like(w) for w in weights]
            v_w = [np.zeros_like(w) for w in weights]
            m_b = [np.zeros_like(b) for b in biases]
            v_b = [np.zeros_like(b) for b in biases]

            for i in range(epochs):
                t = i + 1

                A = [X]
                Z = []

                for j in range(num_layers):
                    z_curr = A[j] @ weights[j] + biases[j]
                    Z.append(z_curr)
                    a_curr = z_curr if j == num_layers - 1 else f5.relu(z_curr)
                    A.append(a_curr)

                A3 = A[-1]

                loss = f5.loss_mse(A3, y)

                self.history.append(loss)

                if i % 1 == 0 or i == epochs:
                    print(f"Iteration {i}, Loss: {loss:.5f}")

                dws = [None] * num_layers
                dbs = [None] * num_layers

                dZ = (2 / m) * (A3 - y)
                dws[-1] = A[-2].T @ dZ
                dbs[-1] = np.sum(dZ, axis=1, keepdims=True)

                for j in reversed(range(num_layers - 1)):
                    dA = dZ @ weights[j + 1].T
                    dZ = f5.d_relu(Z[j]) * dA   # نضرب في مشتقة الـ ReLU للطبقة الحالية
                    dws[j] = A[j].T @ dZ     # A[j] هو الدخل القادم للطبقة الحالية
                    dbs[j] = np.sum(dZ, axis=1, keepdims=True)

                for j in range(num_layers):
                    # أ) تحديث الأوزان (Weights)
                    m_w[j] = beta1 * m_w[j] + (1 - beta1) * dws[j]
                    v_w[j] = beta2 * v_w[j] + (1 - beta2) * (dws[j] ** 2)
                    m_w_hat = m_w[j] / (1 - beta1 ** t)
                    v_w_hat = v_w[j] / (1 - beta2 ** t)
                    weights[j] -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)

                    # ب) تحديث الانحياز (Biases)
                    m_b[j] = beta1 * m_b[j] + (1 - beta1) * dbs[j]
                    v_b[j] = beta2 * v_b[j] + (1 - beta2) * (dbs[j] ** 2)
                    m_b_hat = m_b[j] / (1 - beta1 ** t)
                    v_b_hat = v_b[j] / (1 - beta2 ** t)
                    biases[j] -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)


            for  i in range(num_layers):
                self.layers[i].W = weights[i]
                self.layers[i].B = biases[i]


            pass
