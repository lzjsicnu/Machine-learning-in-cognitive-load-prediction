import os, cv2
import numpy as np
from datetime import datetime
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
#import tensorflow.compat.v1 as tf
import tensorflow as tf
#tf.disable_v2_behavior()
#print(tf.__version__) # 注意是2个下划线
#https://download.csdn.net/download/sinat_31337047/10919502
#安装tf-nightly版本

# !pip install -U tensorboard-plugin-profile
# !pip install -q -U keras-tuner
# import kerastuner as kt
import keras_tuner as kt
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops





data_dir = 'image/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

labels = []
inputfirst = tf.placeholder(tf.float32, [None, 640, 480, 1])
input_ = tf.placeholder(tf.float32, [None, 640,480, 1])
with tf.Session() as sess:
    for i, each in enumerate(classes, 1):
        #print("Starting {} images".format(each))
        class_path = data_dir + each
        #print('class_path:',class_path)
        files = os.listdir(class_path)

        for ii, file in enumerate(files, 1):
            #print(os.path.basename(file))
            image_value = tf.read_file(os.path.join(class_path, file))

            img = tf.image.decode_jpeg(image_value, channels=1)

            tf.global_variables_initializer()

            img = tf.image.resize_images(img, [640, 480], method=0)
            print(img)

            imgput = tf.reshape(img, [1,640, 480, 1])
            if ((ii == 1) & (i == 1)):
                inputfirst = imgput
            else:

                inputfirst = tf.concat([inputfirst, imgput], axis=0)
            labels.append(each)

    labels = tf.reshape(labels, [-1])
    #print(inputfirst.shape)
    #print(labels.shape)


#把tensor转成numpy：https://blog.csdn.net/bloom8668/article/details/124325052

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
features_all  = sess.run(inputfirst)  # nd_Test就是ndarray类型，可以在此基础上做其他操作

labels_all = sess.run(labels)#把label的tensor转成了numpy
#print(features_all.shape, labels_all.shape)

#可视化图像
#plt.figure(figsize=(14, 4))
plt.figure()
for i in range(5):
  plt.subplot(1, 5, i+1)
  plt.imshow(np.reshape(features_all[i], (640, 480)).T, cmap='Greys')
  plt.xticks([])
  plt.yticks([])


#数据分割
labels_all = np.reshape(labels_all, len(labels_all))
selection = np.logical_or(labels_all == b'0', labels_all ==b'1')
labels = labels_all[selection]
labels[labels == b'0'] = 0
labels[labels == b'1'] = 1
features = features_all[selection]

test_split = 0.2
split_idx = int(len(features) * test_split)



idxs = np.random.permutation(len(features))
train_idx, test_idx = idxs[split_idx:], idxs[:split_idx]
features_train, features_test = features[train_idx], features[test_idx]
labels_train, labels_test = labels[train_idx], labels[test_idx]
#from keras_multi_head import MultiHeadAttention
# from tensorflow.keras.layers import MultiHeadAttention
# from tensorflow.contrib.rnn import MultiHeadAttention
def model_bayes(hp):
  x_in = tf.keras.Input(shape=features_train[0].shape)
  x_r = tf.keras.layers.Reshape((640*480, 1))(x_in)
  #x = tf.keras.layers.MultiHeadAttention(hp.get('att_heads'), 1, attention_axes=None, dropout=hp.get('dropout'))(x_r, x_r)
  #x = MultiHeadAttention(hp.get('att_heads'), 1, attention_axes=None, dropout=hp.get('dropout'))(x_r,x_r)
  #x = MultiHeadAttention(head_num=2)
  x=tf.compat.v1.keras.layers.MultiHeadAttention(hp.get('att_heads'), 1, attention_axes=None, dropout=hp.get('dropout'))(x_r, x_r)

  if hp.get('skip_connection'):
    x = tf.keras.layers.Add()([x_r, x])
  x = tf.keras.layers.Flatten()(x)
  x_out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(inputs=x_in, outputs=x_out)

  model.compile(
      optimizer=tf.keras.optimizers.Adam(hp.get('learning_rate')),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=['accuracy'])
  return model


#log_dir='logsabc/'
# hps = kt.HyperParameters()
#log_dir = 'logs/' + datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir='logs/'
hps = kt.HyperParameters()

hps.Choice(name='att_heads', values=[1, 8, 15, 25], ordered=True)
hps.Float(name='dropout', min_value=0, max_value=0.8, step=0.2)
hps.Choice(name='learning_rate', values=[1e-2, 1e-3], ordered=True)
hps.Choice(name='skip_connection', values=[True, False], ordered=False)
tuner = kt.BayesianOptimization(
    hypermodel=model_bayes,
    objective='val_accuracy',
    max_trials=50,
    hyperparameters=hps,
    directory=log_dir,project_name='Attention_Classification'
)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

tuner.search(features_train, labels_train, epochs=2, validation_split=0.2, callbacks=[stop_early, tboard_callback]) # interrupted because it took too long