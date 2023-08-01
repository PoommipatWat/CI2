import numpy as np
import matplotlib.pyplot as plt
import copy

class NN:
    def __init__(self, layer, learning_rate = 0.1, momentum_rate=0.9):
        self.w, self.delta_w = self.init_weights_dw(layer)
        self.b, self.delta_bias, self.local_gradient = self.init_bias_lg(layer)
    
        self.layer = layer

        self.V = []

        self.momentum_rate = momentum_rate
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_diff(self, x):
        return x * (1 - x)

    def init_weights_dw(self, layer):
        weights = []
        delta_weights = []
        for i in range(1, len(layer), 1):
            weights.append(np.random.rand(layer[i], layer[i-1]))
            delta_weights.append(np.zeros((layer[i], layer[i-1])))
        return weights, delta_weights

    def init_bias_lg(self, layer):
        biases = []
        delta_biases = []
        local_gradientes = [np.zeros(layer[0])]
        for i in range(1, len(layer), 1):
            biases.append(np.random.rand(layer[i]))
            delta_biases.append(np.zeros(layer[i]))
            local_gradientes.append(np.zeros(layer[i]))
        return biases, delta_biases, local_gradientes

    def feed_forward(self, input):
        self.V = [input]
        for i in range(len(self.layer) - 1):
            self.V.append(self.sigmoid(np.dot(self.w[i],self.V[i]) + self.b[i]))

    def back_propagation(self, design_output):
        for i, j in enumerate(reversed(range(1, len(self.layer), 1))):
            if i == 0:
                error = design_output - self.V[j]
                self.local_gradient[j] = error * self.sigmoid_diff(self.V[j])
            else:
                self.local_gradient[j] = self.sigmoid_diff(self.V[j]) * np.sum(self.w[j] * self.local_gradient[j+1])
            self.delta_w[j-1] = (self.momentum_rate * self.delta_w[j-1]) + np.outer(self.learning_rate * self.local_gradient[j], self.V[j-1])
            self.delta_bias[j-1] = (self.momentum_rate * self.delta_bias[j-1]) + self.learning_rate * self.local_gradient[j]
            self.w[j-1] += self.delta_w[j-1]
            self.b[j-1] += self.delta_bias[j-1]
        return np.sum(error**2) / 2 #return SSE
    
    def train(self, input, design_output, Epoch = 1000, L_error = 0.001):
        N = 0
        keep_error = []
        
        while N < Epoch:
            actual_output = []
            er = 0
            for i in range(len(input)):
                self.feed_forward(input[i])
                actual_output.append(self.V[-1])
                er += self.back_propagation(design_output[i])
            er /= len(input)
            keep_error.append(er)
            N += 1
            print(f"Epoch = {N} | AV_Error = {er}")
        plt.plot(keep_error)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()

    def test(self, input, design_output):
        actual_output = []
        for i in input:
            self.feed_forward(i)
            actual_output.append(self.V[-1])
        actual_output = [element[0] for element in actual_output]
        categories = [f"{element}" for element in range(len(design_output))]
        er = 0
        for i in range(len(actual_output)):
            er += np.sum((actual_output[i]-design_output[i])**2) / 2
        er /= len(actual_output)
        print(f"Test_error = {er}")
        bar_width = 0.2
        index = range(len(categories))
        plt.bar(index, np.array(actual_output), bar_width, label='Actual Output', color='b')
        plt.bar([i + bar_width for i in index], np.array(design_output), bar_width, label='Design Output', color='orange')
        plt.bar([j + bar_width for j in [i + bar_width for i in index]], np.array([100 if abs(actual_output[i]-design_output[i])>100 else abs(actual_output[i]-design_output[i]) for i in range(len(actual_output))]), bar_width, label='Error Error', color='r')
        plt.xlabel('Categories')
        plt.ylabel('Output')
        plt.title('Actual Output vs. Design Output')
        plt.xticks([i + bar_width / 2 for i in index], categories)
        plt.legend()
        plt.tight_layout()
        plt.show()
                
def Read_Data(filename = 'Flood_dataset.txt'):
    data = []
    input = []
    design_output = []
    with open(filename) as f:
        for line in f.readlines()[2:]:
            data.append([float(element[:-1]) for element in line.split()])
    data = np.array(data)
    np.random.shuffle(data)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    epsilon = 1e-8  # A very small value to avoid division by zero
    data = (data - min_vals) / (max_vals - min_vals + epsilon)
    for i in data:
        input.append(i[:-1])
        design_output.append(i[-1])
    return input, design_output

def k_fold_varidation(data, k = 10):
    test = []
    train = []
    for i in range(0, k*int(len(data)*k/100), int(len(data)*k/100)):
        test.append(data[i:int(i+len(data)*k/100)])
        train.append(data[:i] + data[int(i+len(data)*k/100):])
    return train, test

if __name__ == "__main__":
    # เตรียมข้อมูล
    input, design_output = Read_Data()
    input_train, input_test = k_fold_varidation(input)
    design_output_train, design_output_test = k_fold_varidation(design_output)

    # สร้างต้นฉบับ NN
    nn = NN([8, 16, 1], learning_rate=0.3, momentum_rate=0.8)

    for i in range(len(input_train)):
        nn_copy = copy.deepcopy(nn)
        nn_copy.train(input_train[i], design_output_train[i], Epoch=500)
        nn_copy.test(input_test[i], design_output_test[i])



