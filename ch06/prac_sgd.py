class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr # 학습률

    '''
    SGD 과정에서 반복해서 불리는 함수
    params: 가중치 매개변수를 저장하는 딕셔너리
    grads: 기울기를 저장하는 딕셔너리
    '''
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]