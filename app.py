from flask import Flask
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.utils  import secure_filename
import os


import numpy as np

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, \
  Conv2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Activation, add
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import load_img, img_to_array





app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload'

rslt = -1
file = ""

@app.route('/')
def index():    
    return render_template('index.html')


@app.route('/result',methods=['POST']) 
def result():
    global rslt,file
    if request.method=='POST':
        print('pd1')
        f = request.files['file1']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        pt = os.path.join(os.getcwd(),app.config['UPLOAD_FOLDER'], filename)
        
        file = filename
        
        IMAGE_SIZE = [224, 224]
        
        i = Input(shape=IMAGE_SIZE + [3])
        x = ZeroPadding2D(padding=(3, 3))(i)
        x = Conv2D(64, (7, 7),
                strides=(2, 2),
                padding='valid',
                kernel_initializer='he_normal'
                )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256])
        x = identity_block(x, 3, [64, 64, 256])

        x = conv_block(x, 3, [128, 128, 512])
        x = identity_block(x, 3, [128, 128, 512])
        x = identity_block(x, 3, [128, 128, 512])
        x = identity_block(x, 3, [128, 128, 512])
        
        
        x = Flatten()(x)
        prediction = Dense(4, activation='softmax')(x)
        
        mod = Model(inputs=i, outputs=prediction)
        
        mod.load_weights('model.hdf5')
        
        #load the image
        my_image = load_img(pt, target_size=(224, 224))

        #preprocess the image
        my_image = img_to_array(my_image)
        my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))

        my_image = preprocess_input2(my_image)

        #make the prediction
        pred = mod.predict(my_image)
        
        rslt = pred.argmax()
        print(file)
        
        return render_template('result.html',res=rslt,filep=file)
    else:
        return "else"



@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


def identity_block(input_, kernel_size, filters):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1),
               kernel_initializer='he_normal'
    )(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, kernel_size, padding='same',
               kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, (1, 1), 
               kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)

    x = add([x, input_])
    x = Activation('relu')(x)
    return x

def conv_block(input_,
               kernel_size,
               filters,
               strides=(2, 2)):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1), strides=strides,
               kernel_initializer='he_normal'
    )(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f2, kernel_size, padding='same',
               kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f3, (1, 1),
               kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(f3, (1, 1), strides=strides,
                      kernel_initializer='he_normal'
    )(input_)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def preprocess_input2(x):
  x /= 127.5
  x -= 1.
  return x