import os
import tensorflow as tf
import numpy as np
import scipy
import cv2
from tensorflow.keras.callbacks import TensorBoard
import sys
from model import MyModel
from dataload import handle_data
from config import num_epochs, learning_rate, batch_size, weight_path, image_shape, train_dir
from dataload import train_generator
from deeplab import DeepLabV3Plus
#from keras.optimizers import Adam

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format("demo"),
    histogram_freq=1, batch_size=32,
    write_graph=True, write_grads=False, write_images=True,
    embeddings_freq=0, embeddings_layer_names=None,
    embeddings_metadata=None, embeddings_data=None, update_freq=500
)

# one check point per epoch
checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path+'fcn_20191021.ckpt',monitor='loss', 
                                                    save_weights_only=True,verbose=1,
                                                    save_best_only=True,save_freq='epoch',mode = 'min')
                                                    

# create training set
train_list_dir = os.listdir(train_dir)
train_list_dir.sort()
train_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32,tf.float32), (tf.TensorShape([None, None, None]),tf.TensorShape([None, None, None])))

train_dataset = train_dataset.shuffle(buffer_size=len(train_list_dir))
train_dataset = train_dataset.batch(batch_size)

#sys.exit()
#model = DeepLabV3Plus(image_shape[0], image_shape[1], nclasses=4)
model = MyModel(4)
model.load_weights(weight_path+'fcn_20191021.ckpt')

#optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=0.0001)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'
)

#mean_IOU = tf.keras.metrics.MeanIoU(num_classes=4)
model.compile(
    optimizer=optimizer,
    loss=tf.compat.v2.nn.softmax_cross_entropy_with_logits,
    metrics=['accuracy']
)


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.compat.v1.Session(config=config)

model.fit(train_dataset, epochs=num_epochs, callbacks=[tensorboard, checkpoint])
model.summary()




