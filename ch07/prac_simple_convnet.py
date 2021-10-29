import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
# from common.functions import *
from collections import OrderedDict
from common.layers import *
from prac_simple_convnet import Convolution, Pooling

class SimpleConvNet:
    '''
    클래스를 초기화한다. 가중치 매개변수도 함께 초기화한다.
    conv - relu - pool - affine - relu - affine - softmax

    input_dim: 입력 데이터(채널 수, 높이, 너비)의 차원（MNIST의 경우엔 784
    conv_param: 합성곱 계층의 하이퍼파라미터 딕셔너리
        filter_num: 필터 수
        filter_size: 필터 크기
        stride: 스트라이드
        pad: 패딩
    hidden_size: 은닉층(완전연결)의 뉴런 수
    output_size: 출력층(완전연결)의 뉴런 수（MNIST의 경우엔 10）
    weight_init_std: 초기화 시 정규분포의 스케일
    '''
    def __init__(self, input_dim=(1,28,28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 가중치 초기화, 학습 매개변수 저장
        self.params = {} # 인스턴스 변수
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['ReLU1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        
        self.last_layer = SoftmaxWithLoss()

    '''
    예측(추론)을 수행한다. 입력 신호를 출력으로 변환한다.
    x: 이미지 데이터
    '''
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    '''
    손실 함수의 값을 구한다. predict()의 결과와 정답 레이블을 바탕으로 교차 엔트로피 오차를 구한다.
    x: 입력 데이터
    t: 정답 레이블
    '''
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t) # SoftmaxWithLoss에는 Softmax와 CEE가 포함됨

    '''
    신경망의 정확도를 구한다.
    x: 입력 데이터
    t: 정답 레이블
    '''
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

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
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse() # 레이어 반대로 (역전파를 위함)
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}                              # 기울기 딕셔너리
        grads['W1'] = self.layers['Conv1'].dW   # 1번째 층의 가중치의 기울기
        grads['b1'] = self.layers['Conv1'].db   # 1번째 층의 가중치의 기울기
        grads['W2'] = self.layers['Affine1'].dW # 2번째 층의 가중치의 기울기
        grads['b2'] = self.layers['Affine1'].db # 2번째 층의 편향의 기울기
        grads['W3'] = self.layers['Affine2'].dW # 3번째 층의 가중치의 기울기
        grads['b3'] = self.layers['Affine2'].db # 3번째 층의 편향의 기울기

        return grads