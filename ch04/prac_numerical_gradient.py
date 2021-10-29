import numpy as np

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    # 각 원소에 대한 수치 미분
    for idx in range(x.size):
        tmp_val = x[idx] # 값 저장
        # f(x + h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x - h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        # 수치 미분
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 값 복원 (다음 loop를 위함)

    return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis = 1)

print(numerical_gradient(function_2, np.array([3.0, 4.0]))) # [6. 8.]
print(numerical_gradient(function_2, np.array([0.0, 2.0]))) # [0. 4.]
print(numerical_gradient(function_2, np.array([3.0, 0.0]))) # [6. 0.]