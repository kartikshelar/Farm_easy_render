from flask import Flask, render_template, request, Markup,jsonify, request,redirect
import numpy as np
import pandas as pd
import requests
import config
import pickle
import torch

from torchvision import transforms
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import io
from utils_py_files.model import ResNet9
from utils_py_files.disease_new import disease_dic
from utils_py_files.fertilizer import fertilizer_dic
from utils_py_files.fertilizer_chemical_based import fert_dic

from flask import Flask, render_template, url_for, redirect, current_app

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

import os

crop_recommendation_model_path = 'models/RandomForest.pkl'
fert_recommendation_model_path = 'models/Fert_classifier.pkl'
#disease_model_path= "disease_model.pkl"
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

fert_recommendation_model = pickle.load(
    open(fert_recommendation_model_path, 'rb'))

#disease_model = pickle.load(
#    open(disease_model_path, 'rb'))

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_classes_new= ['Apple Scab Leaf',
                      'Apple leaf',
                      'Apple rust leaf',
                      'Bell_pepper leaf',
                      'Bell_pepper leaf spot',
                      'Blueberry leaf',
                      'Cherry leaf',
                      'Corn Gray leaf spot',
                      'Corn leaf blight',
                      'Corn rust leaf',
                      'Peach leaf',
                      'Potato leaf',
                      'Potato leaf early blight',
                      'Potato leaf late blight',
                      'Raspberry leaf',
                      'Soyabean leaf',
                      'Squash Powdery mildew leaf',
                      'Strawberry leaf',
                      'Tomato Early blight leaf',
                      'Tomato Septoria leaf spot',
                      'Tomato leaf',
                      'Tomato leaf bacterial spot',
                      'Tomato leaf late blight',
                      'Tomato leaf mosaic virus',
                      'Tomato leaf yellow virus',
                      'Tomato mold leaf',
                      'Tomato two spotted spider mites leaf',
                      'grape leaf',
                      'grape leaf black rot']

# disease_model_path = 'models/plant_disease_model.pth'
# disease_model = ResNet9(3, len(disease_classes))
# disease_model.load_state_dict(torch.load(
#     disease_model_path, map_location=torch.device('cpu')))
# disease_model.eval()

disease_model= torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt') 


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = "e16d265fdb13046429aad7621a032f7b"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    #print(city_name)
    complete_url = "https://api.openweathermap.org/data/2.5/weather?q="+city_name+"&appid="+api_key
    
    response = requests.get(complete_url)
    x = response.json()
    #print(x)
    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        #print(temperature,"+",humidity)
        return temperature, humidity
    else:
        return None
    
# def showimage(myimage):
#     if (myimage.ndim>2):  #This only applies to RGB or RGBA images (e.g. not to Black and White images)
#         myimage = myimage[:,:,::-1] #OpenCV follows BGR order, while matplotlib likely follows RGB order
         
#     fig, ax = plt.subplots(figsize=[10,10])
#     ax.imshow(myimage, cmap = 'gray', interpolation = 'bicubic')
#     plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#     plt.show()
    
# def bgremove2(myimage):
#     # First Convert to Grayscale
#     myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
 
#     ret,baseline = cv2.threshold(myimage_grey,127,255,cv2.THRESH_TRUNC)
 
#     ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)
 
#     ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)
 
#     foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
#     # Convert black and white back into 3 channel greyscale
#     background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
 
#     # Combine the background and foreground to obtain our final image
#     finalimage = background+foreground
#     return finalimage
    
# def predict_image(img, model=disease_model):
#     """
#     Transforms image to tensor and predicts disease label
#     :params: image
#     :return: prediction (string)
#     """
#     print("here")
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.ToTensor(),
#     ])
#     image = Image.open(io.BytesIO(img))
#     img_t = transform(image)
#     img_u = torch.unsqueeze(img_t, 0)
#     print("here_2")
#     # Get predictions from model
#     yb = model(img_u)
#     # Pick index with highest probability
#     _, preds = torch.max(yb, dim=1)
#     prediction = disease_classes[preds[0].item()]
#     # Retrieve the class label
#     return prediction

def predict_image(img, model= disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    print("here")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    
    #img_t = transform(image)
    #img_u = torch.unsqueeze(img_t, 0)
    results= model(image) 
    
    df = results.pandas().xyxy[0]
    labels=df["name"][0]
    #print(labels)
    return labels

    
app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///New_database.db'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
#postgres://crop_recommend_database_user:fEAFBbmsacpdBdQADkgqqgiyScmQFiSe@dpg-cgj7vtubb6mo06k1ed50-a.oregon-postgres.render.com/crop_recommend_database
app.config['SECRET_KEY'] = 'thisisasecretkey'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
 
class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    register = SubmitField('Register')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')

with app.app_context():
    db.create_all()
    
@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            print()
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                
                return redirect(url_for('crop'))
    return render_template('login.html', form=form)

@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()   
    
    if form.validate_on_submit():
        print("validate")
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        print("user added")
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/crop', methods=['GET', 'POST'])
def crop():
    return render_template('crop.html')

@ app.route('/predict', methods=['POST'])
def crop_prediction():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    
    N = int(final_features[0][0])
    P = int(final_features[0][1])
    K = int(final_features[0][2])
    ph = float(final_features[0][3])
    rainfall = float(final_features[0][4])
    city = final_features[0][6]
    
    if weather_fetch(city) != None:
        temperature, humidity = weather_fetch(city)
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        print(final_prediction)
        return render_template('crop-result.html', prediction=final_prediction)
    
    else:

        return render_template('try_again.html')

@ app.route('/fertilizer-chemical')
def fertilizer_recommendation_chemical():
    

    return render_template('fertilizer-chemical.html')
    
@ app.route('/fertilizer-organic')
def fertilizer_recommendation_organic():
    

    return render_template('fertilizer-organic.html')

# @app.route('/disease-predict', methods=['GET', 'POST'])
# def disease_prediction():
    

#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)

#         file = request.files.get('file')
#         if not file:
#             return render_template('disease.html')
#         try:

#             img = file.read()
            
#             prediction = predict_image(img)

#             prediction = Markup(str(disease_dic[prediction]))
#             return render_template('disease-result.html', prediction=prediction)
#         except:
#             pass
#     return render_template('disease.html')

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get('file')
        if not file:
            return render_template('disease.html')
        try:

            img = file.read()
            prediction = predict_image(img)
            print(prediction)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction)
        except:
            pass
    return render_template('disease.html')
   
# render fertilizer recommendation result page


@ app.route('/fertilizer-predict-chemical', methods=['POST']) 
def fert_recommend():
    crop_type = str(request.form['cropname'])

    N = int(request.form['nitrogen'])
    K = int(request.form['pottasium'])
    P = int(request.form['phosphorous'])

    soil_type = str(request.form['soilname'])
    moisture = int(request.form['moisture'])
    
    city = str(request.form['city'])
    
    if soil_type == "Black":
        st= 0
    elif soil_type == "Clayey":
        st= 1
    elif soil_type == "Loamy":
        st= 2
    elif soil_type == "Red":
        st= 3
    elif soil_type == "Sandy":
        st= 4
        
    if crop_type == "Barley":
        ct= 0
    elif crop_type == "Cotton":
        ct= 1
    elif crop_type == "Ground Nuts":
        ct= 2
    elif crop_type == "Maize":
        ct= 3
    elif crop_type == "Millets":
        ct= 4
    elif crop_type == "Oily seeds":
        ct= 5
    elif crop_type == "Paddy":
        ct= 6
    elif crop_type == "Pulses":
        ct= 7
    elif crop_type == "Sugarcane":
        ct= 8
    elif crop_type == "Tabacco":
        ct= 9
    elif crop_type == "Wheat":
        ct= 10
    
    print(ct)
    if weather_fetch(city) != None:
        temperature, humidity = weather_fetch(city)
        data_fert = np.array([[temperature,humidity,moisture,st,ct,N, K, P]])
        #print(data_fert)
        my_prediction = fert_recommendation_model.predict(data_fert)
        if my_prediction[0] == 0:
            key ="10-26-26"
        elif my_prediction[0] ==1:
            key ="14-35-14"
        elif my_prediction[0] == 2:
            key ="17-17-17"
        elif my_prediction[0] == 3:
            key ="20-20"
        elif my_prediction[0] == 4:
            key ="28-28"
        elif my_prediction[0] == 5:
            key ="DAP"
        else:
            key ="Urea"

        response = Markup(str(fert_dic[key]))
        return render_template('fertilizer-result.html', recommendation=response)
    
    else:

        return render_template('try_again_fert.html')


def fert_recommend_2():
    
    
    crop_type = str(request.form['cropname'])

    N = int(request.form['nitrogen'])
    K = int(request.form['pottasium'])
    P = int(request.form['phosphorous'])

    soil_type = str(request.form['soilname'])
    moisture = int(request.form['moisture'])
    
    city = str(request.form['city'])
    
    if soil_type == "Black":
        st= 0
    elif soil_type == "Clayey":
        st= 1
    elif soil_type == "Loamy":
        st= 2
    elif soil_type == "Red":
        st= 3
    elif soil_type == "Sandy":
        st= 4
        
    if crop_type == "Barley":
        ct= 0
    elif crop_type == "Cotton":
        ct= 1
    elif crop_type == "Ground Nuts":
        ct= 2
    elif crop_type == "Maize":
        ct= 3
    elif crop_type == "Millets":
        ct= 4
    elif crop_type == "Oily seeds":
        ct= 5
    elif crop_type == "Paddy":
        ct= 6
    elif crop_type == "Pulses":
        ct= 7
    elif crop_type == "Sugarcane":
        ct= 8
    elif crop_type == "Tabacco":
        ct= 9
    elif crop_type == "Wheat":
        ct= 10
    
    if weather_fetch(city) != None:
        temperature, humidity = weather_fetch(city)
        data_fert = np.array([[temperature,humidity,moisture,st,ct,N, K, P]])
        print(data_fert)
        my_prediction = fert_recommendation_model.predict(data_fert)
        if my_prediction[0] == 0:
            final_prediction_fert ="10-26-26"
        elif my_prediction[0] ==1:
            final_prediction_fert ="14-35-14"
        elif my_prediction[0] == 2:
            final_prediction_fert ="17-17-17"
        elif my_prediction[0] == 3:
            final_prediction_fert ="20-20"
        elif my_prediction[0] == 4:
            final_prediction_fert ="28-28"
        elif my_prediction[0] == 5:
            final_prediction_fert ="DAP"
        else:
            final_prediction_fert ="Urea"

        print(final_prediction_fert)
        return render_template('fertilizer-result-new.html', prediction=final_prediction_fert)
    
    else:

        return render_template('try_again_fert.html')
       
@ app.route('/fertilizer-predict-organic', methods=['POST']) 
def fert_recommend_organic():
    
   

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response)
        

if __name__ == '__main__':      
    app.run(debug=False)