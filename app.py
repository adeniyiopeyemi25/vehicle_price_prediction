import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.write("""
## Car Price Predition App
         
This application forecasts the price of vehicles in England. To begin, choose the car brand and then provide additional details. The selected vehicle information will be displayed below, and you can observe the predicted price changing as you modify the information.        
         """)
st.write('---')
df = pd.read_csv("clean_adverts.csv")
X, y = df.drop(columns='price'), df['price']

def search_model(car):
    result = df[df['standard_make'].str.contains(car)]
    return result['standard_model'].unique().tolist()

def search_body(car):
    result = df[df['standard_model'].str.contains(car)]
    return result['body_type'].unique().tolist()
def search_fuel(car):
    result = df[df['standard_model'].str.contains(car)]
    return result['fuel_type'].unique().tolist()

st.sidebar.header('Set Vehicle Information Here')
def user_input(): 
    car = st.sidebar.selectbox(label="Brand", options=df['standard_make'].unique().tolist())
    models = {car: search_model(car) for car in df['standard_make'].unique().tolist()}
    if car in models.keys():   
        model = st.sidebar.selectbox(label="Model", options=models[car]) 
    # body = {model: search_body(car) for model in df['standard_model'].unique().tolist()}
    # if model in body.keys():
    #     body = st.sidebar.selectbox(label="Body Type", options=body[model]) 
        body = st.sidebar.selectbox(label="Body Type", options=df['body_type'].unique().tolist())
        fuel = st.sidebar.selectbox(label="Fuel Type", options=df['fuel_type'].unique().tolist())
        age = st.sidebar.slider("Vehicle Age", 1,30,1)
        rounded_age = int(round(age))
        mileage = st.sidebar.slider("mileage", 0,200000,10)

    data = {"standard_make":car,
            'standard_model':model,
            "body_type":body,
            'fuel_type':fuel,
            "vehicle_age": age,
            "mileage":mileage,
            "average_mileage":age * 10000
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input()
enc =pickle.load(open('target.pkl','rb'))
df = enc.transform(input_df)
df = df[["mileage","standard_make","standard_model","body_type","fuel_type","vehicle_age","average_mileage"]]

st.write(""" 
### Your Specified Vehicle Information
          """ )
st.write(input_df.iloc[:,:-1])
st.write('---')
model = pickle.load(open("car_price_model.pkl","rb"))
pred = model.predict(df)

#st.subheader("Prediction")
st.write(f"Predicted Price is: £{round(pred[0],2)}")

# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature Importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')
