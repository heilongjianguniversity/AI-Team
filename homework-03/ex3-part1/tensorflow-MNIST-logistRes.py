
#以下函数的使用方法，请参考：https://tensorflow.google.cn/api_docs/python/

from tensorflow.examples.tutorials.mnist import input_data   # 导入 tensorflow 中的 input_data 子模块，目的是为了后续的导入读取数据
import tensorflow as tf # 导入 tensorflow 库，并且重名为 tf, 便于后面的简写 tf 
import numpy as np  # 导入 numpy 库，并且重名为 np, 便于后面的简写 np

#基本参数设置
batchSize = 30   #batchsize的大小，代表每次训练载入的图像张数
lr = 0.005       #学习率的大小，若后面启用learning rate decay策略，则该值为学习率的初始值
iter = 1000000   #训练的迭代次数
saveInter = 100  #保存结果的频率，即每训练100次保存一次模型训练参数及模型性能
sample_size = 55000  #学习example的总大小，MNIST中官方写60000张，实际为55000（训练）+ 5000（校验），本例中只使用了55000 train

# 对模型输出的结果进行评判，>0.5为“正”，<0.5为“负”
def predict(X):   # 定义一个函数 predict， 作用是用来进行预测
    num = X.shape[0]  # 通过 shape 属性，得到 X 行的个数
    result = [] # 定义一个空的列表 result ，后面通过 append 的方式，向里面添加元素
    for i in range(num):  # for循环语句， i 从0，1，2, 到 num -1
        if X[i]>0.5: # 如果 X[i] 大于 0.5
            result.append(1.0) # 将 1.0 添加到列表 result 中
        else: # 否则，X[i] 小于或等于 0.5
            result.append(0.0)  # 将 0.0 添加到列表 result 中
    return result # 返回 result 的结果

# 加载数据集，建议提前到官网上下载MNIST数据集，并解压到./MNIST文件夹下
# MNIST下载地址：http://yann.lecun.com/exdb/mnist/
def loadData(): # 定义一个 loadData 函数
    file = "../MNIST" # 数据集 MINIST 
    mnist = input_data.read_data_sets(file, one_hot=True)  # input_data.read_data_sets 读取数据
    return mnist # 返回读取的数据 mnist

# 申请模型输入输出的占位符
def create_placeholder(n_x=784,n_y=0): # 定义一个 create_placeholder  函数
    X = tf.placeholder(tf.float32,shape=[None,n_x],name='X')   # 调用tf.placeholder函数，tensorflow 中定义 X
    Y = tf.placeholder(tf.float32, shape=[None,], name='Y')  # 调用tf.placeholder函数，tensorflow 中定义 Y
    return X,Y  #返回 X 和 Y　的数值

# 定义参数，W,b
def initialize_parameters(): # 定义一个 initialize_parameters 函数
    W = tf.Variable({0})  #调用tf.Variable函数，设置模型参数W，W的维度为[784,1]，且初始化为0
    b = tf.Variable({0})  #调用tf.Variable函数，设置模型参数b，b的维度为[1  ,1],且初始化为0
    parameters={'W': W,  # 参数权重 W
                'b': b}  # 参数偏置 b
    return parameters  # 返回参数

# 将标签转换为one-hot形式，本例中未用到该函数，是因为tensorflow中封装了one-hot功能
def convert_one_hot(Y,C):  # 定义一个 convert_one_hot 函数
    one_hot=np.eye(C)[Y.reshape(-1)].T  # 初始化 one_hot 为对角矩阵
    return one_hot  # 返回 one_hot 

# 定义网络模型
def forward_propagation(X,parameters):  # 定义一个 forward_propagation 函数
    W = parameters['W']  # 参数权重 W 
    b = parameters['b']  # 参数偏置 b

    Z1={0}  #调用tensorflow函数，实现Z1=X*W+b
    A1={0}  #调用tf.nn.sigmoid，实现A1 = sigmoid(Z1)
    A1 = tf.clip_by_value({0})  #调用clip_by_value，将A1进行裁剪，使其在[0.001，1.0]之间，是为了避免出现接近于0的极小值，输入np.log()中出现nan的情况
    return A1 # 返回 A1

# 定义loss function
def compute_cost(y_,y,W):  # 定义一个 compute_cost 函数
    #以下的cross_entropy经过了简单变化，在(1.0-y_)*tf.log(1.0-y)之前乘以0.1，是因为正负样本比例基本上为1：9，严重偏向负样本
    #以下添加了正则，也可以尝试去掉
    cross_entropy = -(1.0/batchSize)*tf.reduce_sum({0}) #调用tf.reduce_sum函数，实现交叉熵函数
    return cross_entropy   # 返回 交叉熵函数 的数值 cross_entropy 

# 模型搭建、训练、存储
def model(mnist,Num): # 定义一个  model 函数
    x,y_ = create_placeholder(784, 0) # 调用 create_placeholder 函数，初始化  x,y_ 
    parameters = initialize_parameters() # 调用 initialize_parameters 函数， 初始化 参数
    A1 = forward_propagation(x, parameters)   # 调用 forward_propagation 函数，实现前向反馈

    #设置learning rate decay策略，随着迭代次数的增加，学习率成指数逐渐减小，减小公式为：decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    global_step = tf.Variable(0)  # 调用  tf.Variable 函数， 初始化 global_step 变量
    learning_rate = tf.train.exponential_decay(lr,global_step,decay_steps=sample_size/batchSize,decay_rate=0.98,staircase=True) # 设置指数衰减的 学习率，调用tf.train.exponential_decay。
    
    cost = compute_cost(y_, A1,parameters['W']) # 调用 compute_cost 函数，计算损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step) # 调用 tf.train.GradientDescentOptimizer 函数， 实现梯度下降的优化
    sess = {0}   #调用tf.InteractiveSession()函数，创建Session
    sess.run({0}) #执行tf.global_variables_initializer()，初始化参数
    
    #利用全部样本对模型进行测试
    testbatchX = mnist.train.images  # 导入 mnist 数据中的训练集 图片
    testbatchY = mnist.train.labels  # 导入 mnist 数据中的训练集 标签
     
    modelLast = []  # 定义一个空的列表 modelLast 
    logName = "./log"+str(Num)+".txt" # 新建文件名为  log"+str(Num)+".txt
    
    #保存模型，且设定保存最大迭代次数的4个
    saver = tf.train.Saver(max_to_keep=4)  # 调用 tf.train.Saver 函数，保存模型
    pf = open(logName, "w") # 以 写入的方式 打开文件  log"+str(Num)+".txt
    for i in range(iter): # for 循环结构， 遍历　iter
        #加载minibatch=50个训练样本
        batch = mnist.train.next_batch(batchSize) # 调用  mnist.train.next_batch 函数，复制给 batch
        batchX = batch[0] # 赋值给 batchX 为 batch 中第一个元素
        batchY = batch[1] # 赋值给 batchY为 batch 中第二个元素
        #执行训练
        train_step.run(feed_dict={0})  #执行tensor流图，并为其添加输入x: batchX, y_: batchY[:,Num]

        #每隔saveInter次迭代，保存当前模型的状态，并测试模型精度
        if i % saveInter == 0:  #条件判断语句 if， 如果 i 整除 iter
            [total_cross_entropy,pred,Wsum,lrr] = sess.run([cost,A1,parameters['W'],learning_rate],feed_dict={x:batchX,y_:batchY[:,Num]}) # 调用 sess.run， 启动 tensoflow
            pred1 = predict(pred)  # 调用 predict 函数，进行预测
            
            #保存当前模型的学习率lr、在minibatch上的测试精度
            print('lr:{:f},train Set Accuracy: {:f}'.format(lrr,(np.mean(pred1 == batchY[:,Num]) * 100))) # 输出训练集的准确率等
            pf.write('lr:{:f},train Set Accuracy: {:f}\n'.format(lrr,(np.mean(pred1 == batchY[:,Num]) * 100))) # 写入训练集的准确率
 
            #保存迭代次数、cross entropy
            print("handwrite: %d, iterate times: %d , cross entropy:%g"%(Num,i,total_cross_entropy)) # 输出迭代次数，交叉熵损失函数等
            pf.write("handwrite: %d, iterate times: %d , cross entropy:%g, W sum is: %g\n" %(Num,i,total_cross_entropy,np.sum(Wsum))) # 写入出迭代次数，交叉熵损失函数等
            
            #保存当前参数状态、测试testbatch上的精度
            [testpred] = sess.run([A1],feed_dict={x: testbatchX, y_: testbatchY[:, Num]})  # 调用 sess.run， 启动 tensoflow
            testpred1 = predict(testpred)   # 调用 predict 函数，进行预测
            print('predict sum is: {:f},Testing Set Accuracy: {:f}\n'.format(np.sum(testpred1),(np.mean(testpred1 == testbatchY[:, Num]) * 100)))  # 输出测试集的准确率等
            pf.write('predict sum is: {:f},Testing Set Accuracy: {:f}\n'.format(np.sum(testpred1),(np.mean(testpred1 == testbatchY[:,Num]) * 100))) # 写入测试集的准确率等
            pf.write("\n") # 写入换行字符
            
            #保存当前模型
            saveName = "model/my-model-" + str(Num) # 保存模型为 "model/my-model-" + str(Num)
            saver.save(sess, saveName, global_step=i) # 调用  saver.save 函数，保存模型
            pf.write("save model completed\n") # 写入 save model completed
            
            #若交叉熵出现nan（出现极值），此时停止训练，保存最新的一次模型名称
            if total_cross_entropy != total_cross_entropy: # 条件判断语句 if ， 如果 total_cross_entropy 不等于 total_cross_entropy
                print("is nan, stop") # 输出 is nan, stop
                pf.write("is nan, stop\n") # 写入 is nan, stop
                modelLast = "model/my-model-" + str(Num)+str(i-saveInter) # 模型文件名为  "model/my-model-" + str(Num)+str(i-saveInter)
                break; # break 跳出循环
    pf.close() # close 关闭打开的文件 
    return modelLast # 返回 modelLast  
    
# 模型测试
def test_model(): # 定义 test_model 函数
    mnist = loadData() # 调用 loadData 函数， 导入数据 
    classNum = 10 # 类别 初始化赋值为 10 ， 共有 10 类
    modelNames = [] # 定义一个空的列表 modelNames
    logName = "./logModelNames.txt" #  文件名为 logModelNames.txt
    pf = open(logName, "w") # 以写入的方式打开  logModelNames.txt
    
    #循环训练每个类别与其他类别的二分类器，保存10个分类器模型
    for i in range(classNum): # for 循环语句， 遍历所有 classNum的类别， 
        modelNames.append(model(mnist,i)) # 通过 append 的方式， 向 modelNames 里面添加 model(mnist,i)
        pf.write(modelNames[i]) # 写入 modelNames[i]
        pf.write("\n") # 写入 换行字符
    pf.close() # 关闭文件

if __name__ == '__main__': # 主程序
    test_model() # 调用 test_model 函数 
    

