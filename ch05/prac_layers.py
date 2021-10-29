import numpy as np
from common.functions import softmax, cross_entropy_error

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0) # 0 이하인 원소를 True, 아닌 원소를 False로 정의한 mask
        out = x.copy() # 입력한 np array
        out[self.mask] = 0 # self.mask에서 True인 원소를 0으로 정의
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0 # self.mask에서 True인 원소를 0으로 정의
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.b = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실
        self.y = None    # Softmax의 출력
        self.t = None    # 정답 레이블(원-핫 벡터)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

# py = Relu()
# x = np.array([[1.0, -0.5], [-2.0, 3.0]])
# py.forward(x)