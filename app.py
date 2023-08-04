import re
import PIL
from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import sass

sass.compile(dirname=('static/scss',
             'static/css'), output_style='compressed')

app = Flask(__name__)


labels = {0: 'Mild', 1: 'Moderate', 2: 'No DR',
          3: 'Proliferate DR', 4: 'Severe'}

model = load_model('retina_weights.hdf5')
model_covid = load_model('covid_detection.hdf5')

model.make_predict_function()
model_covid.make_predict_function()


def predict_covid(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    i = image.img_to_array(img)
    i = np.expand_dims(i, axis=0)
    prediction = model_covid.predict(i)

    if prediction == 0:
        res = 'Covid Detected'
    else:
        res = 'Your report is normal'

    return res


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(256, 256))
    i = np.asarray(i, dtype=np.float32)
    i = i/255
    i = i.reshape(-1, 256, 256, 3)
    p = model.predict(i)
    p = np.argmax(p)
    return labels[p]


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/detect-covid', methods=['GET', 'POST'])
def detect_covid():
    return render_template('detect-covid.html')


@app.route('/covid-predict-detail', methods=['GET', 'POST'])
def detect_covid_detail():
    return render_template('covid-predict-detail.html')


@app.route('/detect-diabetic-retinopathy', methods=['GET', 'POST'])
def detect_diabetic_retinopathy():
    return render_template('detect-diabetic-retinopathy.html')


@app.route('/dr-predict-detail', methods=['GET', 'POST'])
def detect_diabetic_retinopathy_detail():
    return render_template('dr-predict-detail.html')


@app.route('/predict-covid', methods=['GET', 'POST'])
def get_img_covid():
    if request.method == 'POST':
        img = request.files['my_img']

        img_path = "static/img/detect/covid"+img.filename
        img.save(img_path)

        p = predict_covid(img_path)

        return render_template('detect-covid.html', prediction=p, img_path=img_path)


@app.route('/predict-dr', methods=['GET', 'POST'])
def get_img_dr():
    if request.method == 'POST':
        img = request.files['dr_img']

        img_path = "static/img/detect/retinopathy"+img.filename
        img.save(img_path)

        p = predict_label(img_path)

        return render_template('detect-diabetic-retinopathy.html', prediction=p, img_path=img_path)

@app.errorhandler(404)  # inbuilt function which takes error as parameter
def not_found(e):  # defining not_found function
    return render_template("404.html")

if __name__ == "__main__":
    app.run(port=8000, debug=True)
