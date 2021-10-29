import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    '''
    클래스를 초기화한다. 가중치 매개변수도 함께 초기화한다.
    input_size: 입력층의 뉴런 수
    hidden_size: 은닉층의 뉴런 수
    output_size: 출력층의 뉴런 수
    '''
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}                                                                    # 가중치 딕셔너리
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)      # 1번째 층의 가중치
        self.params['b1'] = np.zeros(hidden_size)                                           # 1번째 층의 편향
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)     # 2번째 층의 가중치
        self.params['b2'] = np.zeros(output_size)                                           # 2번째 층의 편향

    '''
    예측(추론)을 수행한다. 입력 신호를 출력으로 변환한다.
    x: 이미지 데이터
    '''
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    '''
    손실 함수의 값을 구한다. predict()의 결과와 정답 레이블을 바탕으로 교차 엔트로피 오차를 구한다.
    x: 입력 데이터
    t: 정답 레이블
    '''
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    '''
    신경망의 정확도를 구한다.
    x: 입력 데이터
    t: 정답 레이블
    '''
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    '''
    수치 미분 방식으로 가중치 매개변수의 기울기를 구한다.
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
    오차역전파법으로 가중치 매개변수의 기울기를 구한다.
    numerical_gradient(self, x, t)보다 결과를 빠르게 얻을 수 있다.
    x: 입력 데이터
    t: 정답 레이블
    '''
    # def gradient(self, x, t)

# 가중치와 편향
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
net.params['W1'].shape # (784, 100)
net.params['b1'].shape # (100,)
net.params['W2'].shape # (100, 10)
net.params['b2'].shape # (10,)
# 예측 처리
x = np.random.rand(100, 784) # 더미 입력 데이터(100장 분량)
y = net.predict(x)
# 기울기
t = np.random.rand(100, 10) # 더미 정답 레이블(100장 분량)
grads = net.numerical_gradient(x, t) # 기울기 계산
grads['W1'].shape # (784, 100)
grads['b1'].shape # (100,)
grads['W2'].shape # (100, 10)
grads['b2'].shape # (10,)
