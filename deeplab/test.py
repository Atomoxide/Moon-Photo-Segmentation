import tensorflow as tf
import scipy
import cv2
import os
import numpy as np
from PIL import Image

from model import MyModel
from config import num_epochs, learning_rate, batch_size, weight_path, image_shape, test_dir, result_path
from dataload import test_generator
from deeplab import DeepLabV3Plus

#test_dir ='/home/ubuntu/'
# COLORMAP = [[0, 0, 255], [0, 255, 0]]
COLORMAP = [[0, 255, 0],[0, 0, 0],[0,0,255],[255,0,0]] #The former is the background, the latter is the things other than background
cm = np.array(COLORMAP).astype(np.uint8)

def addweight(pred, test_img):
    # add transparent channel on the original picture
    pred = Image.fromarray(pred.astype('uint8')).convert('RGB')

    test_img = test_img[0]
    out = np.zeros(test_img.shape, test_img.dtype)
    cv2.normalize(test_img, out, 0,
                  255, cv2.NORM_MINMAX)
    image = Image.fromarray(out.astype('uint8')).convert('RGB')
    
    #image = Image.blend(image,pred,0.3)
    #return image
    return pred


def write_pred(image, pred, x_names):
    
    pred = pred[0]  # pred's dim:[h, w, n_class]
    x_name = x_names[0]
    pred = np.argmax(pred, axis=2)  # get the largest number of the channel
    pred = cm[pred]  # convert the pixel value to color map

    weighted_pred = addweight(pred, image) 
    
    #weighted_pred.save(os.path.join(result_path,filename.split("/")[-1]))
    weighted_pred.save('result.png')
    print(filename.split("/")[-1]+" finished")


# def write_img(pred_images, filename):

#     pred = pred_images[0]
#     COLORMAP = [[0, 0, 255], [0, 255, 0]]
#     cm = np.array(COLORMAP).astype(np.uint8)

#     pred = np.argmax(np.array(pred), axis=2)

#     pred_val = cm[pred]
#     cv2.imwrite(os.path.join("data",filename.split("/")[-1]), pred_val)
#     print(os.path.join("data",filename.split("/")[-1])+"finished")


test_dataset = tf.data.Dataset.from_generator(
    test_generator, tf.float32, tf.TensorShape([None, None, None]))
test_dataset = test_dataset.batch(5)

model = DeepLabV3Plus(image_shape[0], image_shape[1], nclasses=4)
#model = MyModel(4)
model.load_weights(weight_path+'fcn_20191021.ckpt')

test_list_dir = os.listdir(test_dir)
test_list_dir.sort()
test_filenames = [test_dir + filename for filename in test_list_dir]

for filename in test_filenames:
    image = scipy.misc.imresize(
        scipy.misc.imread(filename), image_shape)  # image dim=[h, w, channel]
    
    #image = image/255
    #image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    #image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    #image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    image = image[np.newaxis, :, :, :].astype("float32")

    out = model.predict(image)  # out-dim =[batch, h, w, n_class]
    write_pred(image, out, filename)
