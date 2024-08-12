import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the linear regression model from the pickle file
with open('Lin_reg.pkl', 'rb') as model_file:
    lr_model = pickle.load(model_file)
# Load the logistic regression model from the pickle file
with open('Log_reg.pkl', 'rb') as model_file:
    log_reg_model = pickle.load(model_file)

# Load the random forest classifier model from the pickle file
with open('SVM.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Define a function to classify the weather type based on the predicted label
def classify(num):
    if num == 0:
        return 'Mostly Sunny - Range of Output: (90 - 100)%'
    elif num == 1:
        return 'Minute Drizzle - Range of Output: (50 - 80)%'
    elif num == 2:
        return 'Foggy Day - Range of Output: (30 - 50)%'
    elif num == 3:
        return 'Heavy Rains Expected - Range of Output: (0 - 20)%'
    elif num == 4:
        return 'Snowy Day - Range of Output: (0 - 5)%'

# Define the main function for the Streamlit app
def main():
    # Add some CSS styles for improved appearance
    st.markdown("""
        <style>
            .title {
                font-size: 36px;
                color: #FF4500;
                text-align: center;
                margin-bottom: 30px;
            }
            .button {
                background-color: #FF4500;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 18px;
                text-align: center;
                cursor: pointer;
            }
            #team-name {
                position: fixed;
                bottom: 10px;
                left: 10px;
                color: #FF4500;
                font-weight: bold;
                font-size: 45px;
            }
            #about-section {
                margin-top: 30px;
                margin-bottom:50px;

            }
            .graph-container {
                width: 10%;
                margin: auto;
           }
            .pop-up-animation {
                animation: pop-up 0.5s ease-out;
            }
            @keyframes pop-up {
                0% {
                    transform: scale(0);
                }
                100% {
                    transform: scale(1);
                }
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Solar Energy Output Prediction")
    st.write("Enter the weather parameters below to predict the weather type.")
    model_selection = st.sidebar.selectbox('Choose Model', ('Linear Regression', 'Logistic Regression', 'SVM Model'))

    
    tmx = st.slider('Maximum Temperature (°C)', 10.0, 40.0)
    tmi = st.slider('Minimum Temperature (°C)', -10.0, 30.0)
    wi = st.slider('Wind Speed (m/s)', 0.0, 10.0)
    
    inputs = [[tmx, tmi, wi]]
    
    if st.button('Predict', key='predict_button'):
        if model_selection == 'Linear Regression':
            prediction = lr_model.predict(inputs)
            predicted_weather_type = classify(int(prediction[0]))
            st.write(f'Predicted Weather Type: {predicted_weather_type}',height = 150 ,class_='pop-up-animation')
        elif model_selection == 'Logistic Regression':
            prediction = log_reg_model.predict(inputs)
            predicted_weather_type = classify(int(prediction[0]))
            st.write(f'Predicted Weather Type: {predicted_weather_type}',height = 150,class_='pop-up-animation')
        elif model_selection == 'SVM Model':
            prediction = svm_model.predict(inputs)
            predicted_weather_type = classify(int(prediction[0]))
            st.write(f'Predicted Weather Type: {predicted_weather_type}',height = 150,class_='pop-up-animation')

    # Add line graphs for each parameter
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Maximum Temperature
    max_temp = np.linspace(10, 40, 50)
    output_max_temp = [lr_model.predict([[temp, tmi, wi]])[0] for temp in max_temp]
    axs[0].plot(max_temp, output_max_temp, color='blue')
    axs[0].set_title('Output vs Maximum Temperature')
    axs[0].set_xlabel('Maximum Temperature (°C)')
    axs[0].set_ylabel('Output (%)')
    
    # Minimum Temperature
    min_temp = np.linspace(-10, 30, 50)
    output_min_temp = [lr_model.predict([[tmx, temp, wi]])[0] for temp in min_temp]
    axs[1].plot(min_temp, output_min_temp, color='green')
    axs[1].set_title('Output vs Minimum Temperature')
    axs[1].set_xlabel('Minimum Temperature (°C)')
    axs[1].set_ylabel('Output (%)')
    
    # Wind Speed
    wind_speed = np.linspace(0, 10, 50)
    output_wind_speed = [lr_model.predict([[tmx, tmi, speed]])[0] for speed in wind_speed]
    axs[2].plot(wind_speed, output_wind_speed, color='red')
    axs[2].set_title('Output vs Wind Speed')
    axs[2].set_xlabel('Wind Speed (m/s)')
    axs[2].set_ylabel('Output (%)')
    
    # Adjust layout
    plt.tight_layout()

    # Display graphs
    st.pyplot(fig)

    # Display team name in the bottom left corner
    st.markdown("<div id='team-name'>OnVizz</div>", unsafe_allow_html=True)

    # About section
    st.markdown("<h2 id='About-Section'>About</h2>", unsafe_allow_html=True)
    st.write("This is the UI for the ML Project that we developed. For the backend we have used three diffrent models and then compare the output for the best two to consider the output. These Models are backed by a huge amount of dataset hence increasing its accuracy to a good amount and can be called reliable. Further changes can be made based on accessing ML Models")

# Run the Streamlit app
if __name__ == '__main__':
    main()
