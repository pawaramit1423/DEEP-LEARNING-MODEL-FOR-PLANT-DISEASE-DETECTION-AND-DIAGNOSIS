import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import os
 

import json



app = Flask(__name__)

@app.route('/analysis', methods=['POST'])
def analysis():
    import base64
    imgstring=request.form['imageData']
    
    imgdata = base64.b64decode(imgstring)
    filename = 'testpicture/some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            str_label = 'healthy'
        elif np.argmax(model_out) == 1:
            str_label = 'bacterial'
        elif np.argmax(model_out) == 2:
            str_label = 'viral'
        elif np.argmax(model_out) == 3:
            str_label = 'lateblight'

        if str_label =='healthy':
            status ="HEALTHY"
        else:
            status = "UNHEALTHY"

        print(status)
        if str_label == 'bacterial':
            diseasename = "Bacterial Spot\n\n"
            #print(diseasename)
            rem = "The remedies for Bacterial Spot are:\n"
            #print(rem)
    
            rem1 = "Discard or destroy any affected plants.\nDo not compost them.\nRotate yoour tomato plants yearly to prevent re-infection next year.\nUse copper fungicites."
            #print(rem1)
            strp="The disease detected is "+diseasename+"."+rem+""+rem1
            print(strp)
            x={"value":str(strp)}
            return json.dumps(x)
            
        elif str_label == 'viral':
            diseasename = "Yellow leaf curl virus.\n\n"
            
            rem = "The remedies for Yellow leaf curl virus are:\n"
    
            rem1 = "Monitor the field, handpick diseased plants and bury them.\nUse sticky yellow plastic traps.\nSpray insecticides such as organophosphates, carbametes during the seedliing stage.\nUse copper fungicites"
            strp="The disease detected is "+diseasename+"\n"+rem+""+rem1+"."
            print(strp)
            x={"value":str(strp)}
            return json.dumps(x)
            
        elif str_label == 'lateblight':
            diseasename = "Late Blight"
            
            rem = "The remedies for Late Blight are:\n\n"
    
            rem1 = "Monitor the field, remove and destroy infected leaves.\nTreat organically with copper spray.\nUse chemical fungicides,the best of which for tomatoes is chlorothalonil."
            strp="The disease detected is "+diseasename+"\n"+rem+""+rem1+"."
            print(strp)
            x={"value":str(strp)}
            return json.dumps(x)
        else:
            print("plant is healthy")
            strp="The plant leaf is healthy."
            x={"value":str(strp)}
            return json.dumps(x)
            

                        
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
