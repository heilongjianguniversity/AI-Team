from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# MNIST数据存放的路径
file = "../MNIST"

# 导入数据，首先检测file路径下是否存在数据集，若不存在，则到网上下载.
# MNIST下载地址：http://yann.lecun.com/exdb/mnist/
# 注意：下载后需要解压
mnist = input_data.read_data_sets(file, one_hot=True)#读取数据集，标签数据设置为one-hot格式。即n维标签中只有一个数据为1，其余为0

# 模型的输入和输出
# 为模型的输入输出申请占位符，作为外部数据与网络模型的交互接口
# 784=28*28
x  = tf.placeholder(tf.float32, shape={0})  #申请占位符 输入图像 N*784的矩阵 [None, 784]
y_ = tf.placeholder(tf.float32, shape={0})  #申请占位符 输入label N*10的矩阵[None, 10]

# 将tensor图中的输入和变量进行计算  通过tf.layers.dense搭建全连接网络层，并为该层设置对应的输入、神经元个数、激活函数
# 通过units设置神经元的个数，通过activation设置激活函数，可设定的激活函数，请参考https://tensorflow.google.cn/api_docs/python/tf/nn/softmax

A1 = tf.layers.dense(inputs=x, units={0},activation=tf.nn.{0})  #{0}为待补充, 添加全连接层，神经元个数为16个，激活函数为sigmoid、tanh或relu
A2 = tf.layers.dense(inputs=A1,units={0},activation=tf.nn.{0})  #{0}为待补充，添加全连接层，神经元个数为16个，激活函数为sigmoid、tanh或relu
y  = tf.layers.dense(inputs=A2,units=10, activation=tf.nn.{0})  #{0}为待补充，添加全连接层，设置激活函数为sigmoid或softmax，由于输出类别是10，所以输出层神经元个数为10

# 交叉熵 用来度量y_与y之间的差异性
# y_表示样本的标签 one-hot形式 ; y表示tensor流图计算出的值，即预测值
cross_entropy = -tf.reduce_sum(y_*tf.log(y))#对损失求和

# 训练 利用梯度下降法，以0.01的学习率最小化目标函数（cross_entropy）
train_step = tf.train.GradientDescentOptimizer({0}).minimize({0}) #设置随机梯度下降的学习率为0.01，最小化目标函数为cross_entropy

# 创建Session，用于启动tensor图
sess = tf.InteractiveSession()

# 调用global_variables_initializer函数，将前面定义的Variable变量按照设置的初始化方式，进行初始化
sess.run({0})  #执行tf.global_variables_initializer()，初始化模型参数

#循环训练，设置迭代次数为10000
for i in range({0}):
    #选取mnist训练数据集，设置minibatchsize为50，即选取样本集中的50个样本
    batch = mnist.train.next_batch({0})
    #启动tensor流图，并执行训练，输入数据为图像（batch[0]）和对应的标签（batch[1]）
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

################################### 测试  ###################################
# 计算模型预测结果与标签中相等的部分
# 调用tf.equal计算模型预测结果y与标签结果y_的差异，预测正确则返回1，预测错误则返回0；
# tf.argmax(y, 1)为计算y中每行数据最大值的索引;
correct_prediction = tf.equal(tf.argmax(y, 1), {0})

# 根据correct_prediction计算模型预测精度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 启动tensor流图，计算模型预测精度，模型输入数据为train/test的图像和对应标签
print(sess.run(accuracy, feed_dict={x: mnist.train.images, y_:mnist.train.labels}))#计算模型在训练集上的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images,  y_:mnist.test.labels}))#计算模型在测试集上的准确率

# 结果输出
logFileName = "logText.txt"
logFile = open(logFileName, "w")
logFile.write(str(sess.run(accuracy, feed_dict={x: mnist.train.images, y_:mnist.train.labels})))
logFile.write("\n")
logFile.write(str(sess.run(accuracy, feed_dict={x: mnist.test.images,  y_:mnist.test.labels})))
logFile.close()
