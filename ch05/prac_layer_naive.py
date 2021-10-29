# 곱셈 계층
class MulLayer:
    # 변수 x와 y를 초기화
    def __init__(self):
        self.x = None
        self.y = None
    
    # 순전파
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    # 역전파
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

# 덧셈 계층
class AddLayer:
    # 변수 x와 y를 초기화
    def __init__(self):
        pass

    # 순전파
    def forward(self, x, y):
        out = x + y
        return out
    
    # 역전파
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy