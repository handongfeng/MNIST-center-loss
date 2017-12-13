Here we aim to reproduce the MNIST results from [this paper](https://ydwen.github.io/papers/WenECCV16.pdf) via an implementation in Keras:

*Y. Wen, K. Zhang, Z. Li, and Y. Qiao, ‘A Discriminative Feature Learning Approach for Deep Face Recognition’, in Computer Vision – ECCV 2016, 2016, pp. 499–515.*

Content:

- Results
- Links

**Results**

First we train a classifier using cross-entropy loss alone (basic_MNIST.py):

<img src='./results/epoch-49-basic-train.png' width='400px'/> <img src='./results/epoch-49-basic-val.png' width='400px'/>


We then add the 'center loss term' (centerloss_MNIST.py):

<img src='./results/epoch-49-lambda-1e-05-train.png' width='400px'/> <img src='./results/epoch-49-lambda-1e-05-val.png' width='400px'/>
<img src='./results/epoch-49-lambda-0.0001-train.png' width='400px'/> <img src='./results/epoch-49-lambda-0.0001-val.png' width='400px'/>
<img src='./results/epoch-49-lambda-0.001-train.png' width='400px'/> <img src='./results/epoch-49-lambda-0.001-val.png' width='400px'/>
<img src='./results/epoch-49-lambda-0.01-train.png' width='400px'/> <img src='./results/epoch-49-lambda-0.01-val.png' width='400px'/>
<img src='./results/epoch-49-lambda-0.1-train.png' width='400px'/> <img src='./results/epoch-49-lambda-0.1-val.png' width='400px'/>

---

**Links**

The original authors implement in [Caffe](http://caffe.berkeleyvision.org/):

https://github.com/ydwen/caffe-face

Some other repos which are trying to do similar:

https://github.com/shamangary/Keras-MNIST-center-loss-with-visualization (Keras) <br />
Warning: Not sure if this is the same method as the paper presents

https://github.com/EncodeTS/TensorFlow_Center_Loss (TensorFlow) <br />
Warning: Readme visualizations are on training set

https://github.com/jxgu1016/MNIST_center_loss_pytorch (PyTorch) <br />
Warning: Readme visualizations are on training set