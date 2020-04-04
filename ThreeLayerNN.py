import numpy as np


# this simple neural network is used for binary classification with 3 neurons


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Neural_Net:

    def __init__(self, x, y, label, step_size):
        num = np.random.normal(size=9)
        self.x = x
        self.y = y
        self.step_size = step_size
        self.label = label
        self.a1 = num[0]  # weights
        self.a2 = num[1]  #
        self.a3 = num[2]  #
        self.b1 = num[3]  #
        self.b2 = num[4]  #
        self.b3 = num[5]  #
        self.c1 = num[6]  # Bias
        self.c2 = num[7]  #
        self.c3 = num[8]  #
        # forward pass
        self.z1 = self.a1 * self.x + self.b1 * self.y + self.c1
        self.z2 = self.a2 * self.x + self.b2 * self.y + self.c2
        self.n1 = sigmoid(self.z1)
        self.n2 = sigmoid(self.z2)
        self.output = 0
        self.sign = 0

    def forwardPass(self):
        # assigning initialized randoms to weights and biases
        # z = sum of all weights * input + bias
        # a1,a2,a3 b1,b2,b3 == weights c1,c2,c3 == bias

        # pushing z through activation function(sigmoid)
        self.z1 = self.a1 * self.x + self.b1 * self.y + self.c1
        self.z2 = self.a2 * self.x + self.b2 * self.y + self.c2
        self.n1 = sigmoid(self.z1)
        self.n2 = sigmoid(self.z2)
        self.output = self.a3 * self.n1 + self.b3 * self.n2

        # Optimization: maximizing or minimizing to fit proper classification

        self.sign = 1 if self.label == 1 and self.output < 1 else -1 if self.label == -1 and self.output > -1 else 0

    def backProp(self):
        # back prop(partial derivatives) or chain rule
        dz1 = self.a3 * self.n1 * (1 - self.n1)
        da1, db1, dc1 = dz1 * self.x, dz1 * self.y, dz1

        dz2 = self.b3 * self.n2 * (1 - self.n2)
        da2, db2, dc2 = dz2 * self.x, dz2 * self.y, dz2

        da3, db3, dc3 = self.n1, self.n2, 1
        # adding random small number
        self.a1 = self.a1 + self.sign * da1 + self.step_size
        self.a2 = self.a2 + self.sign * da2 + self.step_size
        self.a3 = self.a3 + self.sign * da3 + self.step_size
        self.b1 = self.b1 + self.sign * db1 + self.step_size
        self.b2 = self.b2 + self.sign * db2 + self.step_size
        self.b3 = self.b3 + self.sign * db3 + self.step_size
        self.c1 = self.c1 + self.sign * dc1 + self.step_size
        self.c2 = self.c2 + self.sign * dc2 + self.step_size
        self.c3 = self.c3 + self.sign * dc3 + self.step_size

    # function to check math output of neural network
    def Final_Stats(self):
        print(f'x: {self.x}, y: {self.y}, label: {self.label}, output: {self.output}, sign: {self.sign}')

    def train(self, iterations, training_data, training_labels):
        for i in range(iterations):
            num = np.random.randint(training_data.shape[0])
            self.x, self.y = training_data[num]
            self.label = training_labels[num]
            self.forwardPass()
            self.backProp()

    def predict(self, data, labels):
        for i in range(data.shape[0]):
            self.x, self.y = data[i]
            curr_labels = labels[i]
            self.z1 = self.a1 * self.x + self.b1 * self.y + self.c1
            self.z2 = self.a2 * self.x + self.b2 * self.y + self.c2
            self.n1 = sigmoid(self.z1)
            self.n2 = sigmoid(self.z2)
            self.output = self.a3 * self.n1 + self.b3 * self.n2
            prediction = 0
            if self.output >= 1:
                prediction = 1
            if self.output <= -1:
                prediction = -1
            print(f'DataPoint: {i + 1}, actual label: {curr_labels}, Predicted label: {prediction}')
   
if __name__ == "__main__":
    
    
    data = np.array([[0.0, 0.7],
                 [-0.3, -0.5],
                 [3.0, 0.1],
                 [-0.1, -1.0],
                 [-1.0, 1.1],
                 [2.1, -3.0]])

    labels = np.array([1, -1, 1, -1, -1, 1])
    num = np.random.randint(data.shape[0])
    x, y = data[num]
    label = labels[num]

    nn = Neural_Net(x, y, label, step_size=0.01)
    nn.train(iterations=1000, training_data=data, training_labels=labels)
    nn.predict(data, labels)
    
    

