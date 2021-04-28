# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 23:45:32 2021

@author: Pulkit_PC
"""

from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("ticketprice.pkl", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Date_of_Journey
        date_dep = request.form["Dep_time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        # print("Journey Date : ",Journey_day, Journey_month)

        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # print("Departure : ",Dep_hour, Dep_min)

        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # print("Arrival : ", Arrival_hour, Arrival_min)

        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)
        # print("Duration : ", dur_hour, dur_min)

        # Total Stops
        Total_stops = int(request.form["stops"])
        # print(Total_stops)

        # Airline
        # AIR ASIA = 0 (not in column)
        airline=request.form['airline']
        if(airline=='Air India'):
            Air_India = 1
            Indigo = 0
            Go_Air = 0
            SpiceJet = 0
            Vistara = 0
            Trujet = 0 
            Air_Asia = 0
            

        elif (airline=='Indigo'):
            Air_India = 0
            Indigo = 1
            Go_Air = 0
            SpiceJet = 0
            Vistara = 0
            Trujet = 0 
            Air_Asia = 0
            
        elif (airline=='Go Air'):
            Air_India = 0
            Indigo = 0
            Go_Air = 1
            SpiceJet = 0
            Vistara = 0
            Trujet = 0 
            Air_Asia = 0
            
        elif (airline=='SpiceJet'):
            Air_India = 0
            Indigo = 0
            Go_Air = 0
            SpiceJet = 1
            Vistara = 0
            Trujet = 0 
            Air_Asia = 0
            
        elif (airline=='Vistara'):
            Air_India = 0
            Indigo = 0
            Go_Air = 0
            SpiceJet = 0
            Vistara = 1
            Trujet = 0 
            Air_Asia = 0
            
        elif (airline=='Trujet'):
            Air_India = 0
            Indigo = 0
            Go_Air = 0
            SpiceJet = 0
            Vistara = 0
            Trujet = 1
            Air_Asia = 0

        elif (airline=='Air Asia'):
            Air_India = 0
            Indigo = 0
            Go_Air = 0
            SpiceJet = 0
            Vistara = 0
            Trujet = 0 
            Air_Asia = 1

       

        else:
            Air_India = 0
            Indigo = 0
            Go_Air = 0
            SpiceJet = 0
            Vistara = 0
            Trujet = 0 
            Air_Asia = 0
       
        Source = request.form["Source"]
        if (Source == 'Mumbai'):
            s_Mumbai = 1
            s_New_Delhi = 0
            s_Ahmedabad = 0
            s_Hyderabad = 0
            s_Bengaluru = 0

        elif (Source == 'New Delhi'):
            s_Mumbai = 0
            s_New_Delhi = 1
            s_Ahmedabad = 0
            s_Hyderabad = 0
            s_Bengaluru = 0

        elif (Source == 'Ahmedabad'):
            s_Mumbai = 0
            s_New_Delhi = 0
            s_Ahmedabad = 1
            s_Hyderabad = 0
            s_Bengaluru = 0

        elif (Source == 'Hyderabad'):
            s_Mumbai = 0
            s_New_Delhi = 0
            s_Ahmedabad = 0
            s_Hyderabad = 1
            s_Bengaluru = 0
        
        elif (Source == 'Bengaluru'):
            s_Mumbai = 0
            s_New_Delhi = 0
            s_Ahmedabad = 0
            s_Hyderabad = 0
            s_Bengaluru = 1

        else:
            s_Mumbai = 0
            s_New_Delhi = 0
            s_Ahmedabad = 0
            s_Hyderabad = 0
            s_Bengaluru = 0

        # print(s_Delhi,
        #     s_Kolkata,
        #     s_Mumbai,
        #     s_Chennai)

        # Destination
        # Banglore = 0 (not in column)
        Source = request.form["Destination"]
        if (Source == 'New Delhi'):
            d_New_Delhi = 1
            d_Bengaluru = 0
            d_Ahmedabad = 0
            d_Hyderabad = 0
            d_Mumbai = 0
        
        elif (Source == 'Bengaluru'):
            d_New_Delhi = 0
            d_Bengaluru = 1
            d_Ahmedabad = 0
            d_Hyderabad = 0
            d_Mumbai = 0

        elif (Source == 'Ahmedabad'):
            d_New_Delhi = 1
            d_Bengaluru = 0
            d_Ahmedabad = 0
            d_Hyderabad = 0
            d_Mumbai = 0

        elif (Source == 'Hyderabad'):
            d_New_Delhi = 0
            d_Bengaluru = 0
            d_Ahmedabad = 0
            d_Hyderabad = 1
            d_Mumbai = 0
            
        elif (Source == 'Mumbai'):
            d_New_Delhi = 0
            d_Bengaluru = 0
            d_Ahmedabad = 0
            d_Hyderabad = 0
            d_Mumbai = 1

        else:
            d_New_Delhi = 0
            d_Bengaluru = 0
            d_Ahmedabad = 0
            d_Hyderabad = 0
            d_Mumbai = 0

        prediction=model.predict([[
            Total_stops,
            Journey_day,
            Journey_month,
            Dep_hour,
            Dep_min,
            Arrival_hour,
            Arrival_min,
            dur_hour,
            dur_min,
            Air_India,
            Go_Air,
            Indigo,
            SpiceJet,
            Trujet,
            Vistara,
            Air_Asia,
            s_Mumbai,
            s_New_Delhi,
            s_Ahmedabad,
            s_Hyderabad,
            s_Bengaluru,
            d_New_Delhi,
            d_Bengaluru,
            d_Ahmedabad,
            d_Hyderabad,
            d_Mumbai,
            
        ]])

        output=round(prediction[0],2)

        return render_template('home.html',prediction_text="Your Flight price is Rs. {}".format(output))


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)