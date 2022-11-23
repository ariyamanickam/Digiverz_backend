
from array import array
from turtle import color
from typing import Collection
from urllib import request
from digiverz_portal_API.FlaskRestAPI import test_db
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bson
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import json
from bson import json_util
from flask import jsonify, request
import numpy as np


def model_builder_endpoint(endpoints):
    @endpoints.route("/dqresult", methods=['GET'])
    def find_all_dq_result():
        collection = test_db.dqresults
        user = collection.find() 
        
        
        return json.loads(json_util.dumps(user)) 
    @endpoints.route("/mbresult", methods=['GET'])
    def find_all_people():
        collection = test_db.modelbuilder
        user = collection.find() 
        
        
        return json.loads(json_util.dumps(user))

    @endpoints.route("/modelBuilder", methods=['POST','GET'])
    def model_builder_pickel():

        #*importing the pickel file********

        with open('E:\PythonPractice-main\digiverz_portal_API\saved_steps.pkl', 'rb') as file:
            data = pickle.load(file)
    
        # fig, ax = plt.subplots(1,1, figsize=(12, 7))
        # plt.bar('Salary', 'Country', color='red', width=0.5)
        # plt.suptitle('Salary (US$) v Country')
        # plt.title('')
        # plt.ylabel('Salary')
        # plt.xticks(rotation=90)
        # plt.savefig(r"E:\DIgiVerz-main\src\assests\squares.png", transparent=True)
        
        
        
        #*geting value from the client************

        collection = test_db.modelbuilder
        _req = request.get_json()
        _gender = _req['gender']
        _age = _req['age']
        _height = _req['height']
        _weight = _req['weight']
        _duration = _req['duration']
        _heartrate = _req['heartrate']
        _bodytemp= _req['bodytemp']

        if _gender == "Male":
            _gender_1 = 0
        else:
            _gender_1 = 1
        
        input_data = np.array([[_gender_1, _age, _height,_weight,_duration,_heartrate,_bodytemp ]])


        model = data["model"]
        # le_country = data["le_country"]
        # le_education = data["le_education"]
        # X[:, 0] = le_country.transform(X[:,0])
        # X[:, 1] = le_education.transform(X[:,1])
        print(input_data)
        input_data = input_data.astype(float)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        
        y_pred = model.predict(input_data_reshaped)

        mb_result = np.array_str(y_pred)
        # data = df["Gender"].value_counts()

        # fig1, ax1 = plt.subplots()
        # ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
        # ax1.axis("equal")
        
        # plt.savefig(r"E:\DIgiVerz-main\src\assests\squares1.png",transparent=True)
        # imgmb = {
        #     "imagedata" : bson.Binary(pickle.dumps(fig)),
        #     "contenttype": "image/png"
        # }
        if  request.method == 'POST':
            inserted_id = collection.insert_one({ 'gender': _gender,'age':  _age,'height': _height,'weight': _weight,'duration': _duration,'heartrate': _heartrate,'bodytemp': _bodytemp, 'result':mb_result }).inserted_id
            print(inserted_id)
            resp = jsonify("user inputs added succesfully")
            resp.status_code = 200

            return resp
        return json.loads(json_util.dumps(mb_result))

       
    return endpoints