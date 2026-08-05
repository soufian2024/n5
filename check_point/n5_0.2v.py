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

    def d_sigmoid(Z):
        result = f5.sigmoid(Z) * (1 - f5.sigmoid(Z))
        return result

    def loss_mse(y_predict,y_true):
        mse = np.mean((y_predict - y_true)**2)
        return mse

    def d_loss_mse(y_predict,y_true):
        d_mse = np.mean((y_predict - y_true)*2)
        return d_mse

    def activation(Z,activation='relu'):
        if activation == 'relu':
            A = f5.relu(Z)
        elif activation == 'leaky_relu':
            A = f5.leaky_relu(Z)
        elif activation == 'sigmoid':
            A = f5.sigmoid(Z)

        return A

    def d_activation(Z,activation='relu'):
        if activation == 'relu':
            dA = f5.d_relu(Z)
        elif activation == 'leaky_relu':
            dA = f5.d_leaky_relu(Z)
        elif activation == 'sigmoid':
            dA = f5.d_sigmoid(Z)

        return dA


class nn5:
    class Dense:
        def __init__(self, input_size, output_size, activation='relu'):
            self.W = 2 * (np.random.rand(input_size, output_size) - 0.5)
            self.B = np.zeros((1, output_size))
            self.activation = activation
            self.A = np.array([])
            self.dA = np.array([])

        def forward(self, X):
            self.X = X
            self.Z = np.dot(X, self.W) + self.B

            self.A = f5.activation(self.Z,self.activation)

            return self.A


        def backward(self,da_1,da1):
            self.dA = f5.d_activation(self.Z,self.activation)
            self.da_1 = da_1
            self.da1 = da1

            gw = self.da_1 * self.dA * self.da1
            gb = self.dA * self.da1
            ga = self.W * self.dA * self.da1

            g = [gw,gb,ga]

            return g

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




            # Sou Studio
            # can you make caine from The Amazing Digital Circus with this ? I mean n5,
            # and what about cyn from morder drones ? can you make it ?

            for i in range(epochs):

                for layer in self.layers:
                    X = layer.forward(X)

                loss = f5.loss_mse(self.layers[-1].A,y)
                d_loss = f5.d_loss_mse(self.layers[-1].A,y)

                self.history.append(loss)

                da = d_loss


                for j in range(len(self.layers) - 1):

                    k = -j + len(self.layers) - 1

                    da_1 = self.layers[-k].A

                    p = self.layers[k].backward(da_1,da)

                    da = p[2]

                    self.layers[k].W -= p[0] * lr_init
                    self.layers[k].B -= p[1][0] * lr_init


                    print(p)
                    print('*'*50)


                da_1 = X

                p = self.layers[0].backward(da_1,da)
                self.layers[0].W -= p[0] * lr_init
                self.layers[0].B -= p[1][0] * lr_init

                print(p)
                print('*'*50)
                print(self.history)






            pass
