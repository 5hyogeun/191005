import tensorflow as tf
import numpy as np
class Add:
    def __init__(self):
        pass
    def execute(self, a, b):
        return tf.constant(a) + tf.constant(b)
class Mean:
    def __init__(self):
        pass

    def execute(self):
        x_array = np.arange(18).reshape(3,2,3)
        x2 = tf.reshape(x_array, shape=(-1, 6))
        # 각 열의 합을 계산
        xsum = tf.reduce_sum(x2, axis=0)    # axis = 0 가로
        # 각 열의 평균을 계산
        xmean = tf.reduce_mean(x2, axis=0)

        print('입력 크기 : ', x_array.shape)
        print('크기가 변경된 입력 크기 : \n', x2.numpy())
        print('열의 합 : \n', xsum.numpy())
        print('열의 평균 : \n', xmean.numpy())

if __name__ == '__main__':
    add = Add()
    # print(add.execute(5, 7))
    mean = Mean()
    mean.execute()