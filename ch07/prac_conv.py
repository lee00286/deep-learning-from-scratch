import sys, os
sys.path.append(os.pardir)
from common.util import im2col

# 배치 크기가 1(데이터 1개), 채널 3개, 높이와 너비가 7x7인 데이터
x1 = np.random.rand(1, 3, 7, 7) # (데이터 수, 채널 수, 높이, 너비)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) # (9, 75) 원소 75개

# 배치 크기가 10(데이터 10개), 채널 3개, 높이와 너비가 7x7인 데이터
x2 = np.random.rand(10, 3, 7, 7) # (데이터 수, 채널 수, 높이, 너비)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) # (90, 75) 원소 75개

# 필터는 채널 3개, 5x5 데이터 => 원소 75개

class Convolution:
    '''
    W: 필터(가중치)
    b: 편향
    stride: 스트라이드
    pad: 패딩
    '''
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W # (FN 필터 개수, C 채널, FH 필터 높이, FW 필터 너비)의 4차원 형상
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # 출력 형상
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # 입력 데이터를 im2col로 전개
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 필터를 reshape를 사용해 2차원 배열로 전개
        col_W = self.W.reshape(FN, -1).T # -1은 원소 수가 변환 후에도 똑같이 유지되도록 묶어줌
                                         # (10, 3, 5, 5) 형상의 W는 750개의 원소를 가짐
                                         # reshape(10, -1)를 호출하면 750개의 원소를 10묶음으로 만들어 줌 (10, 75)
        # 행렬의 곱 + 편향
        out = np.dot(col, col_W) + self.b

        # 출력 데이터를 적당한 형상으로 바꿔줌
        # transpose는 다차원 배열의 축 순서를 바꿔줌 (인덱스 지정으로 축의 순서 변경)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 입력 데이터를 im2col로 전개
        col = im2col(x, self.pool_h, self.pool_w, self.stride)
        # 필터를 reshape를 사용해 2차원 배열로 전개
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 각 줄의 최댓값들을 세로로 나열 (axis=1은 행 방향)
        out = np.max(col, axis=1)

        # 출력 데이터를 적당한 형상으로 바꿔줌
        # transpose는 다차원 배열의 축 순서를 바꿔줌 (인덱스 지정으로 축의 순서 변경)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out