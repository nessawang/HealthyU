import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import openai
import time

st.write("# Welcome to HealthyU APP")
    #st.sidebar.success("Select an option from the sidebar")

st.markdown(
        
        """
        #### We have developed models specifically designed to assist in identifying the potential risk factors associated with lung cancer, stroke, and diabetes. While we are not medical professionals, this information can provide valuable insights and increase your awareness of the potential risks involved.
        """
        )
    
video_file = open('final_project_homepage.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes, start_time=0)
#st.title('Health Risk Prediction')
st.markdown('#### To help us make accurate predictions for you, please answer the following questions.')
st.divider()

# Function to display the question and get user input
def ask_question(feature, key=None):
    if key is None:
        key = feature['description']
    
    st.markdown(f'<h2 style="font-size: 22px;; line-height: 1.5;">{feature["name"]}</h2>', unsafe_allow_html=True)
    if feature['type'] == 'select':
       value = st.selectbox("", feature['choices'], key=key)
    
    else:
        value = st.slider("", min_value=feature['min'], max_value=feature['max'], step=feature['step'], value=feature['min'], key=key)
    
    return value


# Function to get user inputs for Lung Cancer prediction
def get_lung_cancer_inputs(age, gender):
    features = [
        {'name': 'Gender','description': 'Gender', 'choices': ['M', 'F'], 'type': 'select'},
        {'name': 'age','description': 'Age', 'min': 0, 'max': 100, 'step': 1, 'type': 'int'},
        {'name': 'Smoke','description': 'Smoking', 'choices': ['N', 'Y'], 'type': 'select'},
        {'name': 'Do you have Chronic Disease?', 'description': 'Chronic Disease', 'choices': ['N', 'Y'], 'type': 'select'},  
        {'name': 'Do you have Allergy?', 'description': 'Allergy', 'choices': ['N', 'Y'], 'type': 'select'},
        {'name': 'Do you think you may have an alcohol-related problem?', 'description': 'Alcohol', 'choices': ['N', 'Y'], 'type': 'select'},
        {'name': 'Are you coughing frequently or consistently?','description': 'Coughing', 'choices': ['N', 'Y'], 'type': 'select'},
        {'name': 'Are you feeling any tightness or pressure in your chest area?','description': 'Chest pain', 'choices': ['N', 'Y'], 'type': 'select'},
        {'name': 'Are you feeling difficulty in breathing?','description': 'SHORTNESS OF BREATH', 'choices': ['N', 'Y'], 'type': 'select'},
        {'name': 'Are you facing any issues with swallowing food or liquids?','description': 'SWALLOWING DIFFICULTY', 'choices': ['N', 'Y'], 'type': 'select'},
        {'name': 'Do you have Yellow fingers?', 'description': 'YELLOW_FINGERS','choices': ['N', 'Y'], 'type': 'select'}]

    
    inputs = {}
    for feature in features:
        if feature['description'] == 'Gender':
            value = gender
        elif feature['description'] == 'Age':
            value = age
        elif feature['description']== 'Smoking':
            value= smoking_status
        else:
            value = ask_question(feature)

        # Map 'N' to 1 and 'Y' to 2 for features with choices ['N', 'Y']
        if 'choices' in feature and feature['choices'] == ['N', 'Y']:
            value = 1 if value == 'N' else 2

        inputs[feature['description']] = value
    return inputs


# Function to predict the result for Lung Cancer
def predict_lung_cancer(inputs):
    model = pickle.load(open('lung_cancer_model.sav', 'rb'))
    df = pd.DataFrame(inputs, index=[0])


    # Show the % prediction
    prediction_percent = model.predict_proba(df)[:, 1] * 100
    st.write("Here is your health risk for getting lung cancer: {:.2f}%".format(prediction_percent[0]))
    return prediction_percent[0]


# Function to get user inputs for Stroke prediction
def get_stroke_inputs(age, gender):
    features = [
        {'name': 'Gender','description': 'Gender', 'choices': ['M', 'F'], 'type': 'select'},
        {'name': 'How old are you?','description': 'Age', 'min': 10, 'max': 100, 'step': 1, 'type': 'int'},
        {'name': 'Height in cm', 'description': 'height', 'min': 70, 'max': 230, 'step': 1, 'type': 'int'},
        {'name': 'Weight in kg', 'description': 'weight', 'min': 30, 'max': 200, 'step': 1, 'type': 'int'},
        {'name': 'Smoke','description': 'smoking_status', 'choices': ['N', 'Y'], 'type': 'select'},
        {'name': 'Do you have heart disease','description': 'heart_disease', 'choices': ['Yes', 'No'], 'type': 'select'},
        {'name': 'Have you ever been married?', 'description': 'ever_married', 'choices' : ['Yes', 'No'], 'type': 'select'},
        {'name': 'What is your blood glucose level?', 'description': 'blood_glucose_level', 'min': 50, 'max': 350, 'step': 10, 'type': 'int'},
        {'name': 'Do you have high blood pressure?', 'description': 'hypertension','choices' : ['Yes', 'No'], 'type': 'select'},
        {'name': 'What type of job do you currently have?', 'description': 'work_type', 'choices': ['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked'], 'type': 'select'}


]
    
    inputs = {}
    for feature in features:
        if feature['description'] == 'Gender':
            value = gender
        elif feature['description'] == 'Age':
            value = age
        elif feature['description'] == 'height':
            value= height
        elif feature['description'] == 'weight':
            value= weight
        elif feature['description']== 'smoking_status':
            value= smoking_status
        elif feature['description']== 'blood_glucose_level':
            value= blood_glucose_level
        elif feature['description']== 'hypertension':
            value = 1 if value == 'Yes' else 0
 
        elif feature['description']=='heart_disease':
            value = 1 if value == 'Yes' else 0
              
        else:
            value = ask_question(feature)

        inputs[feature['description']] = value

    if inputs['height'] > 0:
        inputs['bmi'] = calculate_bmi(inputs['weight'], inputs['height'])
    else: 
        inputs['bmi'] = 19
   
    return inputs


def calculate_bmi(weight, height):
    height_m = height / 100
    bmi = round(weight / (height_m * height_m), 1)
    return bmi


# Function to predict stroke 
def predict_stroke(inputs):
    model = pickle.load(open('stroke_model.sav', 'rb'))
    df = pd.DataFrame(inputs, index=[0])
   
     # Show the % prediction
    prediction_percent = model.predict_proba(df)[:, 1] * 100
    st.write("Here is your health risk for getting stroke: {:.2f}%".format(prediction_percent[0]))
    return prediction_percent[0]


def get_diabetes_inputs(age, gender):
    features = [
        {'name': 'What is your Gender?','description': 'Gender', 'choices': ['Male', 'Female'], 'type': 'select'},
        {'name': 'Please provied your Age','description': 'Age', 'min': 0, 'max': 100, 'step': 1, 'type': 'int'},
        {'name': 'Please provied your height (cm)', 'description': 'height', 'min': 70, 'max': 230, 'step': 1, 'type': 'int'},
        {'name': 'Please provied your weight (kg)', 'description': 'weight', 'min': 30, 'max': 200, 'step': 1, 'type': 'int'},
        {'name': 'What is your blood glucose level?', 'description': 'blood_glucose_level', 'min': 50, 'max': 350, 'step': 10, 'type': 'int'},
        {'name': 'Do you have heart disease','description': 'heart_disease', 'choices': ['Yes', 'No'], 'type': 'select'},
        {'name': 'Do you have high blood pressure?', 'description': 'hypertension','choices' : ['Yes', 'No'], 'type': 'select'}
    
    ]
    
    
    inputs = {}
    for feature in features:
        if feature['description'] == 'Gender':
            value = gender
        elif feature['description'] == 'Age':
            value = age
        elif feature['description'] == 'height':
            value= height
        elif feature['description'] == 'weight':
            value= weight
        elif feature['description']== 'blood_glucose_level':
            value= blood_glucose_level
        elif feature['description']== 'hypertension':
            value = 1 if value == 'Yes' else 0
 
        elif feature['description']=='heart_disease':
            value = 1 if value == 'Yes' else 0
        else:
            value = ask_question(feature)

        inputs[feature['description']] = value

    if inputs['height'] > 0:
        inputs['bmi'] = calculate_bmi(inputs['weight'], inputs['height'])
    else: 
        inputs['bmi'] = 19
   
    return inputs



def predict_diabetes(inputs):
    model = pickle.load(open('diabetes_model.sav', 'rb'))
    df=pd.DataFrame(inputs, index=[0])

    # Show the % prediction
    prediction_procent = model.predict_proba(df)
    if prediction_procent[0][1] in prediction_procent:
        prediction_procent = round(prediction_procent[0][1] * 100, 2)
        st.write("Here is your health risk for getting diabetes: " + str(prediction_procent) + "%")
    
    return prediction_procent

    



# Main code
gender = ask_question({'name': 'What is your Gender?', 'description': 'Gender', 'choices': ['Male', 'Female'], 'type': 'select'})
age = ask_question({'name': 'How old are you?', 'description': 'Age', 'min': 10, 'max': 100, 'step': 1, 'type': 'int'})
height = ask_question({'name': 'Please provied your height (cm)', 'description': 'height', 'min': 70, 'max': 230, 'step': 1, 'type': 'int'})
weight = ask_question({'name': 'Please provied your weight (kg)', 'description': 'weight', 'min': 30, 'max': 200, 'step': 1, 'type': 'int'})
smoking_status = ask_question({'name': 'Do you Smoke?','description': 'smoking_status', 'choices' : ['N', 'Y'], 'type': 'select'})
blood_glucose_level = ask_question({'name': 'What is your blood glucose level?', 'description': 'blood_glucose_level', 'min': 50, 'max': 350, 'step': 10, 'type': 'int'})
hypertension = ask_question({'name': 'Do you have high blood pressure','description': 'hypertension','choices' : ['Yes', 'No'], 'type': 'select'})
heart_disease = ask_question({'name': 'Do you have heart disease?', 'description': 'heart_disease', 'choices' : ['Yes', 'No'], 'type': 'select'})



lung_cancer_inputs = get_lung_cancer_inputs(age, gender)
stroke_inputs = get_stroke_inputs(age, gender)
diabetes_inputs = get_diabetes_inputs(age, gender)



input_data = {
    'Lung Cancer Inputs': [lung_cancer_inputs],
    'Stroke Inputs': [stroke_inputs],
    'Diabetes Inputs': [diabetes_inputs]
}

gpt_input = pd.DataFrame(input_data)


# my openai api key
import secrets

openai.api_key = st.secrets["openai_key"]



# ask ChatGPT for the suggestion    
def ask_GPT(gpt_input):  
    addition1 = """
            I am using datasets from Kaggle.
            https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer ,
            https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset ,
            https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset.
            I trained these models with RandomForestClassifier or KNeighborsClassifier.
            But if you have other better machine learning models, please feel free to use.
            Now I am creating a y_test by letting the user insert thier body condition. 
            y_test :  
                """
    addition2 = """
            Can you base on the inputs from this y_test, predition the health risk by yourself?
            I don't want you just follow these predictions results and give me the answer afterwards.
            I want you base on the anaylsis data, combine with your experience.
            Then, telling me how high(%) is the risk of getting lung cancer, stroke and diabetes. 
            I want your answer will be shown a list of percentage of each disease.
            Please also provied me some suggestions that how to prevent these diseases.
            For example, list the information about diet, exercise, related recipes, and nutritional supplements. give 2 to 3 suggestions to each object.
            And please reply me easy to understand. eapecially for the user that are not familiar with reading long text.
                
                """
    prompt = addition1 + gpt_input.to_string(index=False) + addition2 
 
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a data scientist and a medical consultant. working on a machine learning project."},
                {"role": "user", "content": prompt}
                ],
            )

    return response.choices[0].message.content


if st.button('Predict'):
    lung_cancer_prediction = predict_lung_cancer(lung_cancer_inputs)
    stroke_prediction = predict_stroke(stroke_inputs)
    diabetes_prediction = predict_diabetes(diabetes_inputs)

    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')

    
    chart_data = pd.DataFrame({
        "Prediction": [lung_cancer_prediction, stroke_prediction, diabetes_prediction],
        "Health Condition": ["Lung Cancer", "Stroke", "Diabetes"]
                            })
    
    fig, ax = plt.subplots()

# Customize bar chart properties
    bar_properties = {
        "Lung Cancer": {
            "color": "#cf9db4",  # Customize the color 
            "width": 0.3  # Customize the width 
        },
        "Stroke": {
            "color": "#d0b2a4",  
            "width": 0.3  
        },
        "Diabetes": {
            "color": "#a66f85",  
            "width": 0.3  
        }
    }

    for condition, properties in bar_properties.items():
        ax.bar(condition, chart_data.loc[chart_data["Health Condition"] == condition, "Prediction"], color=properties["color"], width=properties["width"])

    # Set chart labels and title
    ax.set_xlabel("Health Condition")
    ax.set_ylabel("Prediction")
    ax.set_title("Health Risk Prediction")

    # Display the customized bar chart
    st.pyplot(fig)

    st.header("Here also comes your prediction results and suggestions for improving your lifestyle from ChatGPT")
    #st.write("We are going to recommend the following things for you:")
    st.write(ask_GPT(gpt_input))




    




