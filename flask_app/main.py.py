from pyexpat import model
from flask import Flask, render_template
from flask import request
from flask import redirect, url_for
import os
from os import listdir
from os.path import isfile, join
import google
from google.cloud import vision
import io
import pickle
import numpy as np
import pandas as pd
import re
import spacy
import scipy
import sklearn
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io

##--------------------------------##
json_path='C:/Users/ASUS/Desktop/LAMBTON/Capstone Project/flask_app/static/package-buddy-capstone-d9edc5dfd127.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= json_path


app = Flask(__name__)


BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH,'static/models/')

## -------------------- Load Models -------------------
#model_sgd_path = os.path.join(MODEL_PATH,'dsa_image_classification_sgd.pickle')
#scaler_path = os.path.join(MODEL_PATH,'dsa_scaler.pickle')
#model_sgd = pickle.load(open(model_sgd_path,'rb'))
#scaler = pickle.load(open(scaler_path,'rb'))
ner_model_path='C:/Users/ASUS/Desktop/LAMBTON/Capstone Project/flask_app/static/models/PackageBuddy_ner'
model=spacy.load(ner_model_path)

@app.errorhandler(404)
def error404(error):
    message = "ERROR 404 OCCURED. Page Not Found. Please go the home page and try again"
    return render_template("error.html",message=message) # page not found

@app.errorhandler(405)
def error405(error):
    message = 'Error 405, Method Not Found'
    return render_template("error.html",message=message)

@app.errorhandler(500)
def error500(error):
    message='INTERNAL ERROR 500, Error occurs in the program'
    return render_template("error.html",message=message)


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file = request.files['image_name']
        filename = upload_file.filename 
        print('The filename that has been uploaded =',filename)
        # know the extension of filename
        # all only .jpg, .png, .jpeg, PNG
        ext = filename.split('.')[-1]
        print('The extension of the filename =',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            # saving the image
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File saved sucessfully')
            # send to pipeline model
            results = pipeline_model(path_save, model)
            hei = getheight(path_save)
            print(results)
            return render_template('upload.html',fileupload=True,extension=False,data=results,image_filename=filename,height=hei)


        else:
            print('Use only the extension with .jpg, .png, .jpeg')

            return render_template('upload.html',extension=True,fileupload=False)
            
    else:
        return render_template('upload.html',fileupload=False,extension=False)

@app.route('/about/')
def about():
    return render_template('about.html')

def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ =img.shape 
    aspect = h/w
    given_width = 300
    height = given_width*aspect
    return height

def pipeline_model(imagepath, model):
  
  def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    #print('Texts:')

    text=texts[0].description
    
    return(text)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))




  def massage_data(sentence):
    '''Pre process address string to remove new line characters, add comma punctuations etc.'''
    cleansed_sentence1=re.sub(r'(,)(?!\s)',', ',sentence)
    cleansed_sentence2=re.sub(r'(\\n)',', ',cleansed_sentence1)
    cleansed_sentence3=re.sub(r'(?!\s)(-)(?!\s)',' - ',cleansed_sentence2)
    cleansed_sentence4=re.sub(r'\.','',cleansed_sentence3)
    cleansed_sentence= re.sub(r'\n',' ',cleansed_sentence4)
    return(cleansed_sentence)

  def get_txt(path):
    data=detect_text(path)
    data= massage_data(data)
    return(data)

  def Predict(data,model):
    text=[]
    label= []
    
    doc = model(data)
    #print(x)
    for ent in doc.ents:
        text.append(ent.text)
        label.append(ent.label_)
        #print(text)
        dataF=pd.DataFrame(data=[text],columns=label)  
    return(dataF)

  data=get_txt(imagepath)
  output=Predict(data,model)
  return(output)



if __name__ == "__main__":
    app.run(debug=False) 