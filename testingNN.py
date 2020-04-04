from idea.ThreeLayerNN import *

# this neural network is used for binary classification with 3 neurons
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
