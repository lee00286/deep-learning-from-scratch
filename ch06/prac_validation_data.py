# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from common.util import shuffle_dataset

(x_train, t_train), (x_test, t_test) = load_mnist()

# 훈련 데이터를 뒤섞음
x_train, t_train = shuffle_dataset(x_train, t_train)

# 훈련 데이터의 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]    # 검증 데이터
t_val = t_train[:validation_num]    # 검증 데이터
x_train = x_train[validation_num:]  # 훈련 데이터
t_train = t_train[validation_num:]  # 훈련 데이터