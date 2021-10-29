import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

x = np.random.randn(1000, 100)  # 1000개의 데이터를 무작위로 생성
node_num = 100                  # 각 은닉층의 뉴런 수
hidden_layer_size = 5           # 은닉층의 개수
activations = {}                # 활성화값을 저장하는 딕셔너리

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
    
    # 표준편차를 바꿔가면서 실험
    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) # Xavier 초깃값
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x, w) # 가중치와 곱함

    # 활성화 함수를 바꿔가면서 실험
    # z = sigmoid(a)
    z = ReLU(a)
    # z = tanh(a)

    activations[i] = z # 활성화값 저장

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1) # 
    plt.title(str(i + 1) + "-layer")        # 히스토그램 제목
    plt.hist(a.flatten(), 30, range=(0, 1)) # 히스토그램 그리기
plt.show()