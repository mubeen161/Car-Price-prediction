import numpy as np
from flask import Flask,request,jsonify,render_template #request is used for req for home pg, press button, navigate to file// render_template make basic template
import pickle

app=Flask(__name__)
model=pickle.load(open('E:/Udemy Cousre ML/ML in 21 days/regressor.pkl','rb'))
lab=pickle.load(open('E:/Udemy Cousre ML/ML in 21 days/lab','rb'))
lab1=pickle.load(open('E:/Udemy Cousre ML/ML in 21 days/lab1','rb'))
lab2=pickle.load(open('E:/Udemy Cousre ML/ML in 21 days/lab2','rb'))


@app.route('/')#route to home page
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering Results in HTML GUI
    '''
    features=[str(x) for x in request.form.values()]
    year=int(features[0])
    kms_driven=int(features[1])
    final_features=[]
    final_features.append(year)
    final_features.append(kms_driven)
    final_features.append(lab.transform([features[2]])[0])
    final_features.append(lab1.transform([features[3]])[0])
    final_features.append(lab2.transform([features[4]])[0])
    print(len(final_features))
    prediction=model.predict([final_features])
    result=round(prediction[0],2)
    return render_template('index.html',prediction_text='The Price of the car is {}'.format(result))

if __name__=="__main__":
    app.run(debug=True)
