import numpy as np

class Momentum:
    def __init__(self, lr=0.01, momentum=0.09):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    '''
    Momentum 과정에서 반복해서 불리는 함수
    params: 가중치 매개변수를 저장하는 딕셔너리
    grads: 기울기를 저장하는 딕셔너리
    '''
    def update(self, params, grads): 
        if self.v is None:
            # Initialize self.v
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.v[key] * self.momentum - self.lr * grads[key] # 수식 1
            params[key] += self.v[key] # 수식 2

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    
    '''
    Momentum 과정에서 반복해서 불리는 함수
    params: 가중치 매개변수를 저장하는 딕셔너리
    grads: 기울기를 저장하는 딕셔너리
    '''
    def update(self, params, grads):
        if self.h is None:
            # Initialize self.h
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) # 분모가 0이 되지 않도록