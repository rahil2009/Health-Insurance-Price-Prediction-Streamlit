import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("health_insurance_cost_prediction_model.sav",'rb'))

#creating a function for prediction

    #giving a title
st.write("""
            <span style="font-family: Arial, sans-serif; font-size: 33px;">
            <strong>
            Health Insurance Cost Prediction Web App    
            </strong>
            </span>
""", unsafe_allow_html=True)
        # Add a description
st.write("""
            <span style="font-family: Arial, sans-serif; font-size: 18px;">
            <strong>
            In this project, we applied statistical analysis techniques and machine learning methods to predict health insurance costs.
            Our goal was to enhance prediction accuracy and provide valuable insights for pricing models. 
            By experimenting with various regression models, we achieved improved performance. 
            Let’s dive into accurate predictions for health insurance premiums!
            </strong>
            </span>
""", unsafe_allow_html=True)


def health_insurance_cost_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    #print(prediction)

    return prediction[0]

def main():
    
    # #giving a title
    # st.title('Health Insurance Cost Prediction Web App')
    
    #getting input data from user
    
    age = st.number_input("Age of the person",step=1)
    option1 = st.selectbox('Gender',('Male', 'Female')) 
    sex = 1 if option1 == 'Female' else 0
    bmi = st.number_input("BMI",step=0.5)
    children = st.number_input("No. of children",step=1)
    option2 = st.selectbox('Smoker',('Yes', 'No'))
    smoker = 1 if option2 == 'No' else 0
    option3 = st.selectbox('Region',('Southeast', 'Southwest','Northwest','Northeast'))
    if option3 == 'Southeast':
        region = 0
    elif option3 == 'Southwest':
        region = 1
    elif option3 == 'Northwest':
        region = 2
    else:
        region = 3
    

    # code for prediction
    price = ''
    
    #creating a button for Prediction
    if st.button('Predict Health Insurance Cost'):
        price = health_insurance_cost_prediction((age,sex,bmi,children,smoker,region))
        
    st.success('The Predicted Price: '+ str(price)+'$')

    st.text("\n") 
    url = "https://superstore-sales.streamlit.app/"
    st.write(
            "Link to project No:4 [Superstore Sales Reporting and Forecasting Dashboard ](%s)" % url
    )



if __name__ == '__main__':
    main()
    
