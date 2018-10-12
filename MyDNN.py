import numpy as np


class MyDNN:
    def __init__(self, net_structure: np.ndarray, x_set: np.ndarray, y_set: np.ndarray):
        """
        网络结构，用数组表示，
        示例： [2,3,1] layer_0 两个单元，layer_1 三个单元，layer_2 1个单元

        :param net_structure: 各网络层的输入数
        """
        self.weights = []
        self.placeholder = []
        self.net_structure = np.array([])
        self.x_set = np.array([])
        self.y_set = np.array([])
        self.input_length = 0
        self.output_length = 0

        self.set_net_structure(net_structure)
        self.set_dataset(x_set, y_set)

    def set_net_structure(self, net_structure: np.ndarray):
        self.net_structure = net_structure
        for layer in range(len(net_structure)):
            if layer < len(net_structure) - 1:  # 最后一层没有层间权重
                layer_weights = np.random.rand(net_structure[layer] + 1, net_structure[layer + 1])  # 初始化 该层权重形状是该层单元数+1(bias)*下一层单元数
                self.weights.append(layer_weights)
                layer_predict_val = np.ones((net_structure[layer] + 1))  # 初始化，运算值的占位 包括bias项（恒为1）
                self.placeholder.append(layer_predict_val)

            else:
                layer_predict_val = np.ones((net_structure[layer] + 1))  # 初始化，运算值的占位 包括bias项（恒为1）
                self.placeholder.append(layer_predict_val)

        self.input_length = net_structure[0]
        self.output_length = net_structure[-1]

    def set_dataset(self, x_set: np.ndarray, y_set: np.ndarray):
        if len(x_set) == 0 or len(y_set) == 0:
            return
        elif len(x_set[0]) != self.input_length or len(y_set[0]) != self.output_length:
            raise Exception("训练集形状与网络结构不匹配")
        else:
            self.x_set, self.y_set = x_set, y_set

    def fit(self, lr: float = 0.1, epochs: int = 10):
        for epoch_num in range(epochs):
            # print("\nepoch:{}".format(epoch_num))
            epoch_loss = 0
            for step in range(len(self.x_set)):

                # print("\rstep:{}".format(step), end="")
                # 前向传播

                # self.placeholder用于保存网络计算状态
                input_x = self.x_set[step]  # 本次的输入
                predict_y = np.array([])  # 本次的输出

                for layer_num in range(len(self.net_structure)):  # 遍历每层，计算并输出
                    if layer_num == 0:  # 首层赋值为x
                        self.placeholder[layer_num][:-1] = input_x
                    elif layer_num < len(self.net_structure):  # 其余层计算上层的值和权重的点积
                        self.placeholder[layer_num][:-1] = np.dot([self.placeholder[layer_num - 1]], self.weights[layer_num - 1])
                        self.placeholder[layer_num][:-1] = 1 / (1 + np.exp(- self.placeholder[layer_num][:-1]))

                predict_y = self.placeholder[-1][:-1]  # 最后一层保留输出,且不看bias

                # 前向传播完毕, 生成输出 predict_y

                # 反向传播 更新权值
                step_loss = self.cross_entropy_loss(self.y_set[step], predict_y)
                epoch_loss += step_loss
                # print((predict_y, self.y_set[step],step_loss ))
                next_layer_sigma = []  # 存放上一层sigma
                for layer_num in range(len(self.net_structure) - 1, -1, -1):  # 2,1,0
                    if layer_num == 0:  # 0层不用更新
                        continue
                    o = predict_y = self.placeholder[layer_num][:-1]
                    y = truth_y = self.y_set[step] if layer_num == 2 else self.placeholder[layer_num][:-1]
                    x = self.placeholder[layer_num - 1][:]  # 此x 包括bias
                    current_layer_sigma = []
                    for j, oj, yj in zip(range(len(o)), o, y): # 对于每个y 计算 w_j
                        if layer_num == len(self.net_structure) - 1:  # 最后一层
                            rloss_roj = -yj / oj + (1 - yj) / (1 - oj)  # 神经元loss 相对 神经元输出的偏导数 ,该函数值在最后一层由目标函数的导函数算出，其他层由后一层传过去
                        else:
                            try:
                                rloss_roj = np.dot(next_layer_sigma, self.weights[layer_num][j, :])
                            except:
                                print("ERR LOG:layer:{},\nnext_layer_sigma:{},\nself.weights[layer_num][j, :]:{}".format(layer_num,next_layer_sigma,self.weights[layer_num][j, :]))
                                exit(-1)
                        rij_rwj = x  # 神经元输入在w的偏导数
                        roj_rij = oj - oj * oj  # 神经元输出在输入的偏导数,这里是sigmoid函数的导数

                        sigmaj = rloss_roj * roj_rij  # sigma
                        current_layer_sigma.append(sigmaj)
                        rloss_rwij = rloss_roj * rij_rwj * roj_rij
                        # 权重更新
                        self.weights[layer_num - 1][:, j] = self.weights[layer_num - 1][:, j] - lr * rloss_rwij
                    next_layer_sigma.clear()
                    next_layer_sigma.extend(current_layer_sigma)# 下一层sigma使用完毕，赋值，
                    # 权重更新完毕
            epoch_loss /= len(self.x_set)
            print("epoch:{:<5d};\tavg_loss:{}".format(epoch_num, epoch_loss))

    def predict(self, x):
        # 前向传播
        input_x = x  # 本次的输入
        for layer_num in range(len(self.net_structure)):  # 遍历每层，计算并输出
            if layer_num == 0:  # 首层赋值为x
                self.placeholder[layer_num][:-1] = input_x
            elif layer_num < len(self.net_structure):  # 其余层计算上层的值和权重的点积
                self.placeholder[layer_num][:-1] = np.dot([self.placeholder[layer_num - 1]], self.weights[layer_num - 1])
                self.placeholder[layer_num][:-1] = 1 / (1 + np.exp(- np.dot([self.placeholder[layer_num - 1]], self.weights[layer_num - 1])))

        predict_y = self.placeholder[-1][:-1]  # 最后一层保留输出,且不看bias
        return predict_y
        # 前向传播完毕, 生成输出 predict_y

    @staticmethod
    def cross_entropy_loss(predict_y=0, y=0):
        return -y * np.log(predict_y) - (1 - y) * np.log(1 - predict_y)


if __name__ == "__main__":
    x_set = [
        [0.1, 0.2, 0.1],
        [0.2, 0.2, 0.2],
        [0.1, 0.3, 0.3],
        [0.1, 0.4, 0.2],
        [0.4, 0.25, 0.6],
    ]
    y_set = [[0.1 * i[0] + 2 * i[1] * i[2] + 0.02] for i in x_set]
    print(y_set)

    dnn = MyDNN(net_structure=np.array([3, 6, 1]), x_set=np.array(x_set),
                y_set=np.array(y_set))
    dnn.fit(lr=1, epochs=10000)
    print(dnn.predict([0.6, 0.2,0.3]))
    print("truth:{}".format([0.1 * 0.6 + 2 * 0.2 * 0.3 + 0.02]))
    # https://blog.csdn.net/baidu_35570545/article/details/62065343*
    # https://blog.csdn.net/u010859650/article/details/81351829
    # https://blog.csdn.net/zhaomengszu/article/details/77834845*
