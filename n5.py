# ======================================================================
# COPYRIGHT, TRADEMARK AND PROPERTY LICENSE NOTICE
# ======================================================================
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
# ======================================================================


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
import json
import os

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
        # NOTE: full softmax derivative is a Jacobian matrix; returning an
        # elementwise s*(1-s) is a practical approximation often used when
        # softmax is applied element-wise with independent losses. For
        # mathematically correct backprop through softmax, prefer using
        # combined softmax+cross-entropy loss or implement Jacobian-vector
        # product. This is safer than returning ones_like which was incorrect.
        s = f5.softmax(Z)
        return s * (1 - s)

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
        # Derivative corresponding to the tanh-based GELU approximation
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        x = Z
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x**3)
        tanh_val = np.tanh(tanh_arg)
        left = 0.5 * (1.0 + tanh_val)
        # derivative of the tanh-approx term
        right = (0.5 * x * (1 - tanh_val**2) * sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2))
        return left + right

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
        # Computes element-wise partial derivative for Mean Squared Error (averaged)
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
        grad = np.where(is_small_error, error, delta * np.sign(error))
        return grad / y_predict.size

    @staticmethod
    def loss_hinge(y_predict, y_true):
        return np.mean(np.maximum(0, 1 - y_true * y_predict))

    @staticmethod
    def d_loss_hinge(y_predict, y_true):
        grad = np.where(1 - y_true * y_predict > 0, -y_true, 0)
        return grad / y_predict.size

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

    This module provides:
    - Layer base API (forward, backward, update)
    - Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D
    - Simple mechanism for custom layers: any object implementing forward/backward/update
      can be added to Sequential without changing the library code.
    """

    class Layer:
        """Base layer class. Custom layers can subclass this or simply implement
        the same methods (duck typing).
        """
        def forward(self, X):
            raise NotImplementedError
        def backward(self, da):
            raise NotImplementedError
        def update(self, lr):
            pass

    class Dense(Layer):
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
            """Executes linear mapping followed by activation."""
            self.X = X
            self.Z = np.dot(X, self.W) + self.B
            self.A = f5.activation(self.Z, self.activation)
            return self.A

        def backward(self, da1):
            dZ = da1 * f5.d_activation(self.Z, self.activation)
            batch = max(1, self.X.shape[0]) if self.X.ndim >= 2 else 1
            self.dW = np.dot(self.X.T, dZ) / batch
            self.dB = np.sum(dZ, axis=0, keepdims=True) / batch
            ga = np.dot(dZ, self.W.T)
            return [self.dW, self.dB, ga]

        def update(self, lr):
            self.W -= self.dW * lr
            self.B -= self.dB * lr

    class Dropout(Layer):
        def __init__(self, rate=0.5, seed=None):
            self.rate = rate
            self.mask = None
            self.seed = seed

        def forward(self, X, training=True):
            if training:
                rng = np.random.RandomState(self.seed)
                self.mask = (rng.rand(*X.shape) >= self.rate).astype(X.dtype) / (1.0 - self.rate)
                return X * self.mask
            else:
                return X

        def backward(self, da):
            if self.mask is None:
                return da
            return da * self.mask

    class Flatten(Layer):
        def __init__(self):
            self.input_shape = None

        def forward(self, X):
            self.input_shape = X.shape
            return X.reshape(X.shape[0], -1)

        def backward(self, da):
            return da.reshape(self.input_shape)

    class Conv2D(Layer):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, activation='relu'):
            # kernel_size: int (square) for simplicity
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            # Weight shape: (out_channels, in_channels, kh, kw)
            self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
            self.B = np.zeros((out_channels,))
            self.activation = activation
            # caches
            self.X = None
            self.Z = None
            self.dW = None
            self.dB = None

        def _pad(self, X):
            if self.padding == 0:
                return X
            return np.pad(X, ((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)), mode='constant')

        def forward(self, X):
            # X expected shape: (batch, h, w, channels)
            self.X = X
            Xp = self._pad(X)
            batch, h, w, _ = Xp.shape
            kh = kw = self.kernel_size
            out_h = (h - kh) // self.stride + 1
            out_w = (w - kw) // self.stride + 1
            Z = np.zeros((batch, out_h, out_w, self.out_channels))
            for b in range(batch):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = Xp[b, h_start:h_start+kh, w_start:w_start+kw, :]  # shape (kh,kw,in_ch)
                        for oc in range(self.out_channels):
                            Z[b, i, j, oc] = np.sum(patch * self.W[oc].transpose(1,2,0)) + self.B[oc]
            self.Z = Z
            return f5.activation(Z, self.activation)

        def backward(self, da):
            # da shape: (batch, out_h, out_w, out_ch)
            Xp = self._pad(self.X)
            batch, h, w, _ = Xp.shape
            kh = kw = self.kernel_size
            out_h = da.shape[1]
            out_w = da.shape[2]
            dX = np.zeros_like(Xp)
            self.dW = np.zeros_like(self.W)
            self.dB = np.zeros_like(self.B)
            for b in range(batch):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = Xp[b, h_start:h_start+kh, w_start:w_start+kw, :]
                        for oc in range(self.out_channels):
                            grad_out = da[b, i, j, oc]
                            self.dW[oc] += np.transpose(patch, (2,0,1)) * grad_out
                            self.dB[oc] += grad_out
                            dX[b, h_start:h_start+kh, w_start:w_start+kw, :] += (self.W[oc].transpose(1,2,0) * grad_out)
            # remove padding from dX
            if self.padding != 0:
                dX = dX[:, self.padding:-self.padding, self.padding:-self.padding, :]
            # average gradients over batch
            self.dW /= max(1, batch)
            self.dB /= max(1, batch)
            return dX

        def update(self, lr):
            self.W -= self.dW * lr
            self.B -= self.dB * lr

    class MaxPool2D(Layer):
        def __init__(self, pool_size=2, stride=2):
            self.pool_size = pool_size
            self.stride = stride
            self.X = None
            self.argmax = None

        def forward(self, X):
            # X: (batch, h, w, ch)
            self.X = X
            batch, h, w, ch = X.shape
            ph = pw = self.pool_size
            out_h = (h - ph) // self.stride + 1
            out_w = (w - pw) // self.stride + 1
            out = np.zeros((batch, out_h, out_w, ch))
            self.argmax = {}
            for b in range(batch):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = X[b, h_start:h_start+ph, w_start:w_start+pw, :]
                        # max over (ph,pw)
                        out[b, i, j, :] = np.max(patch.reshape(-1, ch), axis=0)
                        # store argmax mask
                        self.argmax[(b,i,j)] = (patch == out[b,i,j,:]).astype(int)
            return out

        def backward(self, da):
            batch, h, w, ch = self.X.shape
            ph = pw = self.pool_size
            out_h = da.shape[1]
            out_w = da.shape[2]
            dX = np.zeros_like(self.X)
            for b in range(batch):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        mask = self.argmax[(b,i,j)]
                        # distribute gradient to max positions
                        grad = da[b, i, j, :]
                        grad_reshaped = grad.reshape(1,1,ch) * mask
                        dX[b, h_start:h_start+ph, w_start:w_start+pw, :] += grad_reshaped
            return dX

    class AvgPool2D(Layer):
        def __init__(self, pool_size=2, stride=2):
            self.pool_size = pool_size
            self.stride = stride
            self.input_shape = None

        def forward(self, X):
            self.input_shape = X.shape
            batch, h, w, ch = X.shape
            ph = pw = self.pool_size
            out_h = (h - ph) // self.stride + 1
            out_w = (w - pw) // self.stride + 1
            out = np.zeros((batch, out_h, out_w, ch))
            for b in range(batch):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = X[b, h_start:h_start+ph, w_start:w_start+pw, :]
                        out[b, i, j, :] = np.mean(patch.reshape(-1, ch), axis=0)
            return out

        def backward(self, da):
            batch, h, w, ch = self.input_shape
            ph = pw = self.pool_size
            out_h = da.shape[1]
            out_w = da.shape[2]
            dX = np.zeros(self.input_shape)
            area = ph * pw
            for b in range(batch):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        grad = da[b, i, j, :]
                        # distribute evenly
                        dX[b, h_start:h_start+ph, w_start:w_start+pw, :] += grad.reshape(1,1,ch) / area
            return dX

    # --- Sequential and persistence (save/load) ---
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
                # Dropout forward signature supports training flag
                if isinstance(layer, nn5.Dropout):
                    current_input = layer.forward(current_input, training=False)
                else:
                    current_input = layer.forward(current_input)
            return current_input

        def train(self, X, y, epochs=100, lr_init=0.01, loss_type='mse'):
            """Executes continuous progressive model parameter backpropagation optimization cycles."""
            print(f"\n🚀 Initiating model optimization across {epochs} epochs...")
            for i in range(epochs):
                current_input = X
                for layer in self.layers:
                    if isinstance(layer, nn5.Dropout):
                        current_input = layer.forward(current_input, training=True)
                    else:
                        current_input = layer.forward(current_input)
                y_pred = current_input
                loss = f5.compute_loss(y_pred, y, loss_type=loss_type)
                self.history.append(loss)
                da = f5.d_compute_loss(y_pred, y, loss_type=loss_type)

                # Execute reverse topological graph traversal for backprop routing
                for layer in reversed(self.layers):
                    gradients = layer.backward(da)
                    # Accept both legacy [dW,dB,ga] or direct ga
                    if isinstance(gradients, list) and len(gradients) == 3:
                        da = gradients[2]
                    else:
                        da = gradients
                    # update if layer has update
                    if hasattr(layer, 'update'):
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
            print(f"{'Layer (type)':<30}{'Output Shape':<20}{'Param #'}")
            print("="*65)

            total_params = 0
            for idx, layer in enumerate(self.layers):
                layer_params = 0
                if hasattr(layer, 'W'):
                    try:
                        layer_params = getattr(layer, 'W').size + getattr(layer, 'B').size
                    except Exception:
                        layer_params = 0
                total_params += layer_params
                lname = layer.__class__.__name__
                layer_name = f"{lname}_{idx+1}"
                output_shape = getattr(layer, 'output_size', '')
                print(f"{layer_name:<30}{str(output_shape):<20}{layer_params:,}")
                print("-"*65)

            print(f"Total trainable params: {total_params:,}")
            print("="*65 + "\n")

        def save(self, filepath="model.n5"):
            """Serializes the entire operational neural graph directly to native .n5 architecture format.

            WARNING: This method uses pickle which will execute arbitrary code when loading.
            Only load .n5 files you trust. For a safer, portable state export use save_state/load_state (.npz).
            """
            if not filepath.endswith('.n5'):
                filepath += '.n5'
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"💾 Dynamic model successfully packed and saved to '{filepath}'!")

        def save_state(self, filepath="model.npz"):
            """Save model parameters (weights, biases and metadata) in a portable .npz archive.

            This method is safer and recommended for exchanging model parameters. It does not
            serialize Python objects or executable code.
            """
            if not filepath.endswith('.npz'):
                filepath += '.npz'

            arrays = {}
            meta = {
                'num_layers': len(self.layers),
                'layers_classes': [layer.__class__.__name__ for layer in self.layers],
                'activations': [getattr(layer, 'activation', None) for layer in self.layers],
                'is_trained': self.is_trained,
                'history': self.history,
            }

            for idx, layer in enumerate(self.layers):
                if hasattr(layer, 'W'):
                    arrays[f'layer{idx}_W'] = layer.W
                if hasattr(layer, 'B'):
                    arrays[f'layer{idx}_B'] = layer.B

            arrays['__meta__'] = np.array(json.dumps(meta), dtype=object)
            np.savez_compressed(filepath, **arrays)
            print(f"💾 Model state saved to '{filepath}' (npz format). Use load_state to restore.")

        @staticmethod
        def load(filepath="model.n5"):
            """Deserializes and initializes instant production-ready .n5 model state binaries.

            WARNING: This uses pickle and will execute code; only load files you trust.
            """
            with open(filepath, 'rb') as f:
                loaded_model = pickle.load(f)
            print(f"📂 Custom .n5 model successfully loaded and ready for immediate deployment!")
            return loaded_model

        @staticmethod
        def load_state(filepath="model.npz"):
            """Load model state saved by save_state. Returns a Sequential instance reconstructed.

            Note: load_state expects numpy.load(..., allow_pickle=True) for __meta__ content.
            Reconstruction will instantiate standard layers (Dense, Conv2D) only. Custom
            or unknown layer classes will not be reconstructed automatically.
            """
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model state file not found: {filepath}")

            data = np.load(filepath, allow_pickle=True)
            meta = json.loads(str(data['__meta__'].tolist()))
            seq = nn5.Sequential()

            for idx in range(meta['num_layers']):
                clsname = meta['layers_classes'][idx]
                activation = meta['activations'][idx]
                # Reconstruct only known layer types: Dense and Conv2D
                if clsname.startswith('Dense') or clsname == 'Dense':
                    W = data[f'layer{idx}_W']
                    B = data[f'layer{idx}_B']
                    layer = nn5.Dense(W.shape[0], W.shape[1], activation=activation)
                    layer.W = W
                    layer.B = B
                    seq.add(layer)
                elif clsname == 'Conv2D':
                    W = data[f'layer{idx}_W']
                    B = data[f'layer{idx}_B']
                    # Infer params from shapes
                    out_ch, in_ch, kh, kw = W.shape
                    layer = nn5.Conv2D(in_channels=in_ch, out_channels=out_ch, kernel_size=kh, activation=activation)
                    layer.W = W
                    layer.B = B
                    seq.add(layer)
                else:
                    print(f"Warning: Can't automatically reconstruct layer type '{clsname}'. Add it manually after load_state.")

            seq.is_trained = bool(meta.get('is_trained', False))
            seq.history = meta.get('history', [])
            print(f"📂 Model state loaded from '{filepath}' into new Sequential instance.")
            return seq


# --- Custom layer example (user can define this in their code and add to Sequential) ---
class SquareLayer(nn5.Layer):
    """Example custom layer: element-wise square activation.

    Forward: y = x**2
    Backward: dy/dx = 2*x * upstream_grad
    """
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return X ** 2

    def backward(self, da):
        # da: upstream gradient
        return 2 * self.X * da


# If run as script show simple examples for each layer
if __name__ == '__main__':
    print('n5 lightweight demo: Dense, Dropout, Conv2D, Pooling, Custom Layer')

    # Dense example
    X = np.random.randn(4, 3)
    y = np.random.randint(0, 2, (4,1))
    model = nn5.Sequential()
    model.add(nn5.Dense(3, 8, activation='gelu'))
    model.add(nn5.Dense(8, 4, activation='leaky_relu'))
    model.add(nn5.Dense(4, 1, activation='sigmoid'))
    model.summary()
    model.train(X, y, epochs=3, lr_init=0.01, loss_type='bce')

    # Dropout example
    X2 = np.random.randn(2,4)
    drop = nn5.Dropout(rate=0.5, seed=42)
    print('\nDropout forward (training):')
    print(drop.forward(X2, training=True))
    print('Dropout backward sample:')
    print(drop.backward(np.ones_like(X2)))

    # Conv2D + Pooling example (toy)
    Xin = np.random.randn(1,8,8,1)  # batch=1, 8x8 grayscale
    conv = nn5.Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, activation='relu')
    out = conv.forward(Xin)
    print('\nConv2D output shape:', out.shape)
    pool = nn5.MaxPool2D(pool_size=2, stride=2)
    p = pool.forward(out)
    print('After MaxPool2D shape:', p.shape)

    # Custom layer demo
    seq2 = nn5.Sequential()
    seq2.add(nn5.Dense(3,3, activation='relu'))
    seq2.add(SquareLayer())
    x3 = np.random.randn(2,3)
    print('\nCustom layer pipeline output:')
    print(seq2.predict(x3))

    # Save/load state demo
    model.save_state('demo_model.npz')
    loaded = nn5.Sequential.load_state('demo_model.npz')
    print('Loaded sequential layers:', [l.__class__.__name__ for l in loaded.layers])
