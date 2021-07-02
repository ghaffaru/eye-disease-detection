from flask import Flask, flash, render_template, request, url_for, redirect, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as im
import numpy as np
import os
import secrets
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = '0ad425f3c8f1b60a0b114d8fcf99a579'

def save_picture(form_picture):
    f_name, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = f_name + f_ext
    picture_path = os.path.join(app.root_path, 'static/pics', picture_fn)
    # output_size = (64,64)
    # i = Image.open(form_picture)
    # i.thumbnail(output_size)
    form_picture.save(picture_path)
    return picture_fn

@app.route('/')
def home():
    return jsonify({'message': 'OK'})
@app.route('/api/classify', methods=['POST'])
def classify():
    if request.files['image']:
        file = request.files['image']
        picture_file = save_picture(file)
        img = im.load_img(os.path.join(app.root_path, 'static/pics', picture_file), target_size=(64, 64))
        img = im.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        model = load_model('model.h5')
        prediction = np.argmax(model.predict(img))

        if prediction == 0:
            value = 'bulging'
        elif prediction == 1:
            value = 'cataract'
        elif prediction == 2:
            value = 'crossed eyes'
        else:
            value = 'glaucoma'
        return jsonify({
            'predicted' : value
        })
    else:
        return jsonify({
            'success': False
        })

#Run server
if __name__ == '__main__':
    app.run(debug=True)