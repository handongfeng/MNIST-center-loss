import basic_MNIST
import centerLoss_MNIST

basic_MNIST.run()
centerLoss_MNIST.run(0.00001)
centerLoss_MNIST.run(0.0001)
centerLoss_MNIST.run(0.001)
centerLoss_MNIST.run(0.01)
centerLoss_MNIST.run(0.1)
centerLoss_MNIST.run(1.0)
