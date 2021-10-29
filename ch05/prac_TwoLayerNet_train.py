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

# 하이퍼파라미터(직접 작성)
iters_num = 10000               # 학습 반복 횟수
train_size = x_train.shape[0]
batch_size = 100                # 미니 배치 크기
learning_rate = 0.1             # lr

# 학습 경과
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

# 학습
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)   # 60,000개의 훈련 데이터
    x_batch = x_train[batch_mask]                           # 훈련 데이터에서 100개의 데이터를 추려냄
    t_batch = t_train[batch_mask]                           # 훈련 데이터에서 100개의 데이터를 추려냄

    # 기울기 계산 - numerical_gradient에서는 loss를 부르고 loss에서는 predict를 부름
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))