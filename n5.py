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
import pickle

class f5:
    """
    Core mathematical engine handling non-linear activation functions,
    their analytical derivatives, and global optimization loss metrics.
    """
    @staticmethod
    def relu(Z):
        return np.where(Z > 0, Z, 0)

    @staticmethod
    def d_relu(Z):
        return np.where(Z > 0, 1, 0)

    @staticmethod
    def leaky_relu(Z):
        return np.where(Z > 0, Z, Z * 0.01)

    @staticmethod
    def d_leaky_relu(Z):
        return np.where(Z > 0, 1, 0.01)

    @staticmethod
    def sigmoid(Z):
        Z = np.clip(Z, -500, 500)  # Bound inputs to eliminate exponential overflow
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def d_sigmoid(Z):
        s = f5.sigmoid(Z)
        return s * (1 - s)

    @staticmethod
    def tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def d_tanh(Z):
        return 1 - np.tanh(Z)**2

    @staticmethod
    def softmax(Z):
        # Shift inputs for numerical stability during exponentiation
        exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

    @staticmethod
    def d_softmax(Z):
        return np.ones_like(Z)

    @staticmethod
    def elu(Z, alpha=1.0):
        return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))

    @staticmethod
    def d_elu(Z, alpha=1.0):
        return np.where(Z > 0, 1, alpha * np.exp(Z))

    @staticmethod
    def selu(Z):
        alpha = 1.673263242354303
        scale = 1.050700987355480
        return scale * np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))

    @staticmethod
    def d_selu(Z):
        alpha = 1.673263242354303
        scale = 1.050700987355480
        return scale * np.where(Z > 0, 1, alpha * np.exp(Z))

    @staticmethod
    def gelu(Z):
        # Fast Gaussian Error Linear Unit approximation via hyperbolic tangents
        return 0.5 * Z * (1 + np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * np.power(Z, 3))))

    @staticmethod
    def d_gelu(Z):
        # Precise continuous derivation of the cumulative distribution function
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * np.power(Z, 3))))
        pdf = np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
        return cdf + Z * pdf

    @staticmethod
    def activation(Z, activation='relu'):
        """Routing multiplexer for standard forward-pass activation functions."""
        if activation == 'relu': return f5.relu(Z)
        elif activation == 'leaky_relu': return f5.leaky_relu(Z)
        elif activation == 'sigmoid': return f5.sigmoid(Z)
        elif activation == 'tanh': return f5.tanh(Z)
        elif activation == 'softmax': return f5.softmax(Z)
        elif activation == 'elu': return f5.elu(Z)
        elif activation == 'selu': return f5.selu(Z)
        elif activation == 'gelu': return f5.gelu(Z)
        return Z

    @staticmethod
    def d_activation(Z, activation='relu'):
        """Routing multiplexer for exact backward-pass gradient computations."""
        if activation == 'relu': return f5.d_relu(Z)
        elif activation == 'leaky_relu': return f5.d_leaky_relu(Z)
        elif activation == 'sigmoid': return f5.d_sigmoid(Z)
        elif activation == 'tanh': return f5.d_tanh(Z)
        elif activation == 'softmax': return f5.d_softmax(Z)
        elif activation == 'elu': return f5.d_elu(Z)
        elif activation == 'selu': return f5.d_selu(Z)
        elif activation == 'gelu': return f5.d_gelu(Z)
        return np.ones_like(Z)

    @staticmethod
    def loss_mse(y_predict, y_true):
        return np.mean((y_predict - y_true)**2)

    @staticmethod
    def d_loss_mse(y_predict, y_true):
        # Computes element-wise partial derivative for Mean Squared Error
        return 2 * (y_predict - y_true) / y_predict.size

    @staticmethod
    def loss_mae(y_predict, y_true):
        return np.mean(np.abs(y_predict - y_true))

    @staticmethod
    def d_loss_mae(y_predict, y_true):
        # Signum-based gradient derivation for L1 absolute optimization paths
        return np.sign(y_predict - y_true) / y_predict.size

    @staticmethod
    def loss_bce(y_predict, y_true):
        y_predict = np.clip(y_predict, 1e-15, 1 - 1e-15)  # Eliminate mathematical log(0) voids
        return -np.mean(y_true * np.log(y_predict) + (1 - y_true) * np.log(1 - y_predict))

    @staticmethod
    def d_loss_bce(y_predict, y_true):
        y_predict = np.clip(y_predict, 1e-15, 1 - 1e-15)
        return (((y_predict - y_true) / (y_predict * (1 - y_predict)))) / y_predict.size

    @staticmethod
    def loss_huber(y_predict, y_true, delta=1.0):
        error = y_predict - y_true
        is_small_error = np.abs(error) <= delta
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        quadratic_loss = 0.5 * (error ** 2)
        return np.mean(np.where(is_small_error, quadratic_loss, linear_loss))

    @staticmethod
    def d_loss_huber(y_predict, y_true, delta=1.0):
        error = y_predict - y_true
        is_small_error = np.abs(error) <= delta
        return np.where(is_small_error, error, delta * np.sign(error)) / y_predict.shape

    @staticmethod
    def loss_hinge(y_predict, y_true):
        return np.mean(np.maximum(0, 1 - y_true * y_predict))

    @staticmethod
    def d_loss_hinge(y_predict, y_true):
        return np.where(1 - y_true * y_predict > 0, -y_true, 0) / y_predict.shape

    @staticmethod
    def compute_loss(y_predict, y_true, loss_type='mse'):
        """Central hub executing requested error assessment loss routines."""
        if loss_type == 'mse': return f5.loss_mse(y_predict, y_true)
        elif loss_type == 'mae': return f5.loss_mae(y_predict, y_true)
        elif loss_type == 'bce': return f5.loss_bce(y_predict, y_true)
        elif loss_type == 'huber': return f5.loss_huber(y_predict, y_true)
        elif loss_type == 'hinge': return f5.loss_hinge(y_predict, y_true)
        return f5.loss_mse(y_predict, y_true)

    @staticmethod
    def d_compute_loss(y_predict, y_true, loss_type='mse'):
        """Central hub computing base error gradient directional matrices."""
        if loss_type == 'mse': return f5.d_loss_mse(y_predict, y_true)
        elif loss_type == 'mae': return f5.d_loss_mae(y_predict, y_true)
        elif loss_type == 'bce': return f5.d_loss_bce(y_predict, y_true)
        elif loss_type == 'huber': return f5.d_loss_huber(y_predict, y_true)
        elif loss_type == 'hinge': return f5.d_loss_hinge(y_predict, y_true)
        return f5.d_loss_mse(y_predict, y_true)


class nn5:
    """
    Object-oriented deep learning structural interface containing discrete layer
    components and progressive sequential execution flow managers.
    """
    class Dense:
        def __init__(self, input_size, output_size, activation='relu'):
            self.input_size = input_size
            self.output_size = output_size
            # Establish initial structural states using standardized He variance bounds
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
            self.B = np.zeros((1, output_size))
            self.activation = activation
            self.A = np.array([])
            self.Z = np.array([])
            self.X = np.array([])
            self.dW = np.array([])
            self.dB = np.array([])

        def forward(self, X):
            """Executes linear matrix mapping followed by activation bounding operations."""
            self.X = X
            self.Z = np.dot(X, self.W) + self.B
            self.A = f5.activation(self.Z, self.activation)
            return self.A

        def backward(self, da1):
            """Applies matrix-level chain rule calculus to isolate discrete parameters errors."""
            dZ = da1 * f5.d_activation(self.Z, self.activation)
            self.dW = np.dot(self.X.T, dZ)
            self.dB = np.sum(dZ, axis=0, keepdims=True)
            ga = np.dot(dZ, self.W.T)  # Transmit backprop error matrix to previous layer
            return [self.dW, self.dB, ga]

        def update(self, lr):
            """Applies standard Stochastic Gradient Descent optimizations to layer components."""
            self.W -= self.dW * lr
            self.B -= self.dB * lr


    class Sequential:
        def __init__(self):
            self.layers = []
            self.history = []
            self.is_trained = False

        def add(self, layer):
            """Appends structural neural layer architecture blocks sequentially."""
            self.layers.append(layer)

        def predict(self, X):
            """Performs dynamic pure forward-pass evaluation loops across existing network arrays."""
            current_input = X
            for layer in self.layers:
                current_input = layer.forward(current_input)
            return current_input

        def train(self, X, y, epochs=100, lr_init=0.01, loss_type='mse'):
            """Executes continuous progressive model parameter backpropagation optimization cycles."""
            print(f"\n🚀 Initiating model optimization across {epochs} epochs...")
            for i in range(epochs):
                y_pred = self.predict(X)
                loss = f5.compute_loss(y_pred, y, loss_type=loss_type)
                self.history.append(loss)
                da = f5.d_compute_loss(y_pred, y, loss_type=loss_type)

                # Execute reverse topological graph traversal for backprop routing
                for layer in reversed(self.layers):
                    gradients = layer.backward(da)

                    # Core Fix: Extract only the backward gradient vector (ga) from index 2
                    da = gradients[2]

                    layer.update(lr_init)

                if (i + 1) % max(1, epochs // 5) == 0 or i == 0:
                    print(f"Epoch {i+1}/{epochs} -> Loss ({loss_type}): {loss:.6f}")

            self.is_trained = True
            print(f"🏁 Training finalized cleanly! Final Loss metric achieved: {self.history[-1]:.6f}")

        def summary(self):
            """Generates a factual diagnostic tracking overview detailing network metrics."""
            status_text = "🟢 [TRAINED & OPTIMIZED]" if self.is_trained else "🔴 [UNTRAINED / RAW ARCHITECTURE]"
            print("\n" + "="*65)
            print(f"Model State: {status_text}")
            print("="*65)
            print(f"{'Layer (type)':<20}{'Output Shape':<20}{'Param #'}")
            print("="*65)

            total_params = 0
            for idx, layer in enumerate(self.layers):
                layer_params = layer.W.size + layer.B.size
                total_params += layer_params
                layer_name = f"Dense_{idx+1} ({layer.activation})"
                output_shape = f"(None, {layer.output_size})"
                print(f"{layer_name:<20}{output_shape:<20}{layer_params:,}")
                print("-"*65)
                print(f"Total trainable params: {total_params:,}")
                print("="*65 + "\n")

        def save(self, filepath="model.n5"):
            """Serializes the entire operational neural graph directly to native .n5 architecture format."""
            if not filepath.endswith('.n5'):
                filepath += '.n5'
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"💾 Dynamic model successfully packed and saved to '{filepath}'!")

        @staticmethod
        def load(filepath="model.n5"):
            """Deserializes and initializes instant production-ready .n5 model state binaries."""
            with open(filepath, 'rb') as f:
                loaded_model = pickle.load(f)
            print(f"📂 Custom .n5 model successfully loaded and ready for immediate deployment!")
            return loaded_model
