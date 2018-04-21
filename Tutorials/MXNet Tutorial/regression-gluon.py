
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import matplotlib.pyplot as plt
import random

# data set
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

# generate random training data set
batch_size = 10
def data_iter():
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i: min(i+batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y, j)

for data, label in data_iter():
    print(data, label)
    break

# build model
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))

# initail model params
net.initialize()
# loss function
square_loss = gluon.loss.L2Loss()
# trainer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# trainning
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0

    for data, label in data_iter():
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))

# print training params with real params
dense = net[0]
print("true_w: %s, trainned_w: %s " % (true_w, dense.weight.data()))
print("true_b: %s, trainned_b: %s " % (true_b, dense.bias.data()))