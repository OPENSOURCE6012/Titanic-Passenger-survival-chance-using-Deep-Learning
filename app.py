import streamlit as st
import pandas as pd
import tensorflow as tf 
import pickle

st.title('Passenger survival chance in titanic journey')
pclass=st.slider('Enter Pclass',1,3,1)
sex=st.selectbox('Enter Sex',('male','female'))
sibsp=st.slider('Enter SibSp',0,8,0)
parch=st.slider('Enter Parch',0,6,0)
fare=st.number_input('enter the fare of passenger')
embarked=st.selectbox('Enter the passenger station from where they started their journey',('southamption','cherbourg','Queenstown'))
data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])

model=tf.keras.models.load_model('model.h5')
with open('label_encoder.pkl','rb') as file:
    label=pickle.load(file)

with open('one_Hot_Encoder.pkl','rb') as file:
    onehot=pickle.load(file)    

    
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)  
data['Sex'] =label.transform(data['Sex'])  
embarked=onehot.transform(data[['Embarked']])
embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out(['Embarked']))
data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)
data[['Pclass','Sex','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','Sex','SibSp','Parch','Fare']])
y=model.predict(data)
y=y[0][0]

def chance(y):
    if y>0.5:
        return "PASSENGER WILL SURVIVE"
    else:
        return "PASSENGER WILL NOT SURVIVE"

if st.button('predict survival chance'):
    st.write('probality of passenger survial chance',y)
    st.write(chance(y))
    