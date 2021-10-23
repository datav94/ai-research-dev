import flask
from flask import request, render_template
from flask_cors import CORS
import joblib

app = flask.Flask('__name__', static_folder='templates', static_url_path='')
CORS(app)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_speices():
    # get form data
    sepallength = float(request.form['sepallengthcm'])
    sepalwidth = float(request.form['sepalwidthcm'])
    petallength = float(request.form['petallengthcm'])
    petalwidth = float(request.form['petalwidthcm'])

    # make an array of that data
    X = [[sepallength, sepalwidth, petallength, petalwidth]]

    # load model
    model = joblib.load('model.pkl')

    # get prediction
    speices = model.predict(X)[0]
    return render_template('predict.html', predict=speices)


if __name__ == '__main__':
    app.run(debug=True)