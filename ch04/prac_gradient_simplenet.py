import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    # Constructor
    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규분포로 초기화
    
    # 예측을 수행
    def predict(self, x):
        return np.dot(x, self.W)

    '''
    손실 함수의 값 계산
    x: 입력 데이터
    t: 정답 레이블
    '''
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss