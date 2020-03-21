# model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Add, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.initializers import Constant
# from tensorflow.nn import conv2d_transpose

from config import image_shape

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights
  

class MyModel(tf.keras.Model):
    def __init__(self, NUM_OF_CLASSESS):
        super().__init__()
        vgg16_model = self.load_vgg()
        self.conv1_1 = vgg16_model.layers[1]
        self.conv1_2 = vgg16_model.layers[2]
        self.pool1 = vgg16_model.layers[3]
        #(128,128)
        self.conv2_1 = vgg16_model.layers[4]
        self.conv2_2 = vgg16_model.layers[5]
        self.pool2 = vgg16_model.layers[6]
        #(64,64)
        self.conv3_1 = vgg16_model.layers[7]
        self.conv3_2 = vgg16_model.layers[8]
        self.conv3_3 = vgg16_model.layers[9]
        self.pool3 =  vgg16_model.layers[10]
        #(32,32)
        self.conv4_1 = vgg16_model.layers[11]
        self.conv4_2 = vgg16_model.layers[12]
        self.conv4_3 = vgg16_model.layers[13]
        self.pool4 =  vgg16_model.layers[14]
        #(16,16)
        self.conv5_1 = vgg16_model.layers[15]
        self.conv5_2 =  vgg16_model.layers[16]
        self.conv5_3 = vgg16_model.layers[17]
        self.pool5 = vgg16_model.layers[18]
        self.conv6 = Conv2D(4096,(7,7),(1,1),padding="same",activation="relu")
        self.drop6 = Dropout(0.5)
        self.conv7 = Conv2D(4096,(1,1),(1,1),padding="same",activation="relu")
        self.drop7 = Dropout(0.5)
        self.score_fr = Conv2D(NUM_OF_CLASSESS,(1,1),(1,1),padding="valid",activation="relu")
        self.score_pool4 = Conv2D(NUM_OF_CLASSESS,(1,1),(1,1),padding="valid",activation="relu")
        self.conv_t1 = Conv2DTranspose(NUM_OF_CLASSESS,(4,4),(2,2),padding="same")
        self.fuse_1 = Add()
        self.conv_t2 = Conv2DTranspose(NUM_OF_CLASSESS,(4,4),(2,2),padding="same")
        self.score_pool3 = Conv2D(NUM_OF_CLASSESS,(1,1),(1,1),padding="valid",activation="relu")
        self.fuse_2 = Add()
        self.conv_t3 = Conv2DTranspose(NUM_OF_CLASSESS,(16,16),(8,8),padding="same", activation="sigmoid", kernel_initializer=Constant(bilinear_upsample_weights(8, NUM_OF_CLASSESS)))
        

    def call(self, input):
      x = self.conv1_1(input)
      x = self.conv1_2(x)
      x = self.pool1(x)
      x = self.conv2_1(x)
      x = self.conv2_2(x)
      x = self.pool2(x)
      x = self.conv3_1(x)
      x = self.conv3_2(x)
      x = self.conv3_3(x)
      x_3 = self.pool3(x)
      x = self.conv4_1(x_3)
      x = self.conv4_2(x)
      x = self.conv4_3(x)
      x_4 = self.pool4(x)
      x = self.conv5_1(x_4)
      x = self.conv5_2(x)
      x = self.conv5_3(x)
      x = self.pool5(x)
      x = self.conv6(x)
      x = self.drop6(x)
      x = self.conv7(x)
      x = self.drop7(x)
      x = self.score_fr(x)  # 第5层pool分类结果
      x_score4 = self.score_pool4(x_4) # 第4层pool分类结果
      x_dconv1 = self.conv_t1(x)  # 第5层pool分类结果上采样
      x = self.fuse_1([x_dconv1,x_score4])  # 第4层pool分类结果+第5层pool分类结果上采样
      x_dconv2 = self.conv_t2(x)  # 第一次融合后上采样
      x_score3 = self.score_pool3(x_3)  # 第三次pool分类
      x = self.fuse_2([x_dconv2,x_score3])  #  第一次融合后上采样+第三次pool分类
      x = self.conv_t3(x)  # 上采样
      return x

    def load_vgg(self):
#        VGG16_weight = "../vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
#        vgg16_model = tf.keras.applications.vgg16.VGG16(weights="../vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, input_tensor=Input(shape=(image_shape[0], image_shape[1], 3)))

        #last_layer = VGG16.output
        
       # set_trainable = False
#        for layer in vgg16_model.layers:
#            if layer.name in ['block1_conv1']:
#                set_trainable = True
#            if layer.name in ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool']:
#                layer.trainable = False
       
        vgg16_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(image_shape[0], image_shape[1], 3)))
        for layer in vgg16_model.layers[:18]:
          layer.trainable = False
        return vgg16_model
