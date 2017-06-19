# -*- coding:utf-8 -*
import numpy as np
import caffe
import os
root = os.path.dirname(__file__)
#bot_data_root = '/home/liuchenfeng/PycharmProjects/vgg_classifier'

# 设置网络结构
net_file = root + '/model/deploy.prototxt'
# 添加训练之后的网络权重参数
caffe_model = root + '/model/VGG_ILSVRC_16_layers.caffemodel'
# 均值文件
mean_file = root + '/model/mean.npy'

mean = np.ones([3,256, 256], dtype=np.float)
mean[0,:,:] = 104
mean[1,:,:] = 117
mean[1,:,:] = 117
mean[2,:,:] = 123

np.save(mean_file, mean)

# 设置使用gpu
caffe.set_mode_gpu()

# 构造一个Net
net = caffe.Net(net_file, caffe_model, caffe.TEST)
# 得到data的形状，这里的图片是默认matplotlib底层加载的
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# matplotlib加载的image是像素[0-1],图片的数据格式[weight,high,channels]，RGB
# caffe加载的图片需要的是[0-255]像素，数据格式[channels,weight,high],BGR，那么就需要转换

# channel 放到前面
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))

# 图片像素放大到[0-255]
transformer.set_raw_scale('data', 255)
# RGB-->BGR 转换
transformer.set_channel_swap('data', (2, 1, 0))
#设置输入的图片shape，1张，3通道，长宽都是224
net.blobs['data'].reshape(1, 3, 224, 224)
# 加载图片
im = caffe.io.load_image(root + '/test/2.jpg')

# 用上面的transformer.preprocess来处理刚刚加载图片
net.blobs['data'].data[...] = transformer.preprocess('data', im)

#输出每层网络的name和shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

# 网络开始向前传播啦
output = net.forward()

# 找出最大的那个概率
output_prob = output['prob'][0]
print '预测的类别是:', output_prob.argmax()

# 找出最可能的前俩名的类别和概率
top_inds = output_prob.argsort()[::-1][:2]
print "预测最可能的前两名的编号: ",top_inds
print "对应类别的概率是: ", output_prob[top_inds[0]], output_prob[top_inds[1]]