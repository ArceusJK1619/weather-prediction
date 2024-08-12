import streamlit as st
import pickle

lr_model = pickle.load(open('Lin_reg.pkl','rb'))


def classify(num):
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


def main():
    st.title("Solar Energy Output Prediction")
    activities=['Linear Regression']
    option=st.sidebar.selectbox('Which ML model would you Like to use',activities)
    st.subheader(option)
    tmx=st.slider('temp_max',10.0,40.0)
    tmi=st.slider('temp_min',-10.0,30.0)
    wi=st.slider('wind',0.0,10.0)
    inputs=[[tmx,tmi,wi]]
    if st.button('Classify'):
        if option=='Linear Regression':
            st.success(classify((lr_model.predict(inputs))))


if __name__=='__main__':
    main()
