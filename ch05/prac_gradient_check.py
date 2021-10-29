import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from prac_TwoLayerNet import TwoLayerNet

# 데이터 읽어오기
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# 클래스 초기화
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 미니 배치
x_batch = x_train[:3] # 시험 레이블
t_batch = t_train[:3] # 정답 레이블

# 기울기 계산
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치 차이의 절댓값의 평균
for key in grad_numerical.keys():
    absolute = np.abs(grad_backprop[key] - grad_numerical[key]) # 가중치 차이의 절댓값
    diff = np.average(absolute)                                 # 절댓값의 평균
    print(key + ":" + str(diff))