import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient
from common.layers import *
from collections import OrderedDict

class TwoLayerNet:
    '''
    클래스를 초기화한다. 가중치 매개변수도 함께 초기화한다.
    input_size: 입력층의 뉴런 수
    hidden_size: 은닉층의 뉴런 수
    output_size: 출력층의 뉴런 수
    weight_init_std: 초기화 시 정규분포의 스케일
    '''
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}                                                                # 신경망의 매개변수를 보관하는 딕셔너리
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 1번째 층의 가중치
        self.params['b1'] = np.zeros(hidden_size)                                       # 1번째 층의 편향
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # 2번째 층의 가중치
        self.params['b2'] = np.zeros(output_size)                                       # 2번째 층의 편향

        # 계층 생성
        self.layers = OrderedDict()                                              # 신경망의 계층을 보관하는 딕셔너리 (순서가 있음)
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])   # 계층 1
        self.layers['Relu1'] = Relu()                                           # 계층 1 활성화 함수
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])   # 마지막 계층
        self.lastLayer = SoftmaxWithLoss()                                      # 마지막 계층 활성화 함수

    '''
    예측(추론)을 수행한다. 입력 신호를 출력으로 변환한다.
    x: 이미지 데이터
    '''
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x) # 각 계층마다 순전파 적용
        return x

    '''
    손실 함수의 값을 구한다. predict()의 결과와 정답 레이블을 바탕으로 교차 엔트로피 오차를 구한다.
    x: 입력 데이터
    t: 정답 레이블
    '''
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t) # SoftmaxWithLoss에는 Softmax와 CEE가 포함됨

    '''
    신경망의 정확도를 구한다.
    x: 입력 데이터
    t: 정답 레이블
    '''
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    '''
    가중치 매개변수의 기울기를 수치 미분 방식으로 구한다.
    x: 입력 데이터
    t: 정답 레이블
    '''
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}                                                  # 기울기 딕셔너리
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 1번째 층의 가중치의 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) # 1번째 층의 편향의 기울기
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) # 2번째 층의 가중치의 기울기
        grads['b2'] = numerical_gradient(loss_W, self.params['b2']) # 2번째 층의 편향의 기울기

        return grads

    '''
    가중치 매개변수의 기울기를 오차역전파법으로 구한다.
    numerical_gradient(self, x, t)보다 결과를 빠르게 얻을 수 있다.
    x: 입력 데이터
    t: 정답 레이블
    '''
    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse() # 레이어 반대로 (역전파를 위함)
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}                              # 기울기 딕셔너리
        grads['W1'] = self.layers['Affine1'].dW # 1번째 층의 가중치의 기울기
        grads['b1'] = self.layers['Affine1'].db # 1번째 층의 편향의 기울기
        grads['W2'] = self.layers['Affine2'].dW # 2번째 층의 가중치의 기울기
        grads['b2'] = self.layers['Affine2'].db # 2번째 층의 편향의 기울기

        return grads