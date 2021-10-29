import numpy as np
from numerical_gradient import numerical_gradient # 편미분 함수

# 변수가 두 개인 함수
def function_2(x):
    return x[0] ** 2 + x[1] ** 2

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x) # 편미분
        x -= lr * grad # 경사법으로 변수의 값 갱신
    
    return x

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.01, step_num=100)) # [-0.39785867  0.53047822]