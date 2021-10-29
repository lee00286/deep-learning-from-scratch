class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg: # 훈련 중일 때
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio   # x와 형상이 같은 배열을 랜덤으로 생성
                                                                        # 랜덤 값 중 dropuout_ratio보다 큰 원소만 True로 설정
                                                                        # 나머지 뉴런(삭제할 뉴런)을 False로 설정
            return x * self.mask
        else: # 시험 중일 때
            return x * (1.0 - self.dropout_ratio)
    
    def backward(self, dout):
        return dout * self.mask # self.mask를 활용하여 순전파 때 신호를 통과시킨 뉴런만 역전파 때도 신호를 통과시킴