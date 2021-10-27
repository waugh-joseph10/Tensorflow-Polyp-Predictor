from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import shutil
import time
import PIL

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(os.path.join(app.root_path, 'uploads/optimal polyp pred.h5'))
image_size = (360, 288)



@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about-us.html')

@app.route('/model-submit', methods=['GET','POST'])
def submit():
    return render_template('model-submit.html')

@app.route('/model-result', methods=['GET','POST'])
def result():
    if request.method == 'POST':
        file = request.files['file']
        file.filename = 'pred-image.jpg'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        img = keras.preprocessing.image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = predictions[0]
        result_msg = ("This image suggests there is a %.2f percent likelihood of cancer." % (100 * (1 - score)))

        return render_template('model-result.html', result_msg = result_msg)

if __name__ == "__main__":
    app.run(debug=True, port=5003)
