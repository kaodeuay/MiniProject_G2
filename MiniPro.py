import streamlit as st
import pandas as pd
import pickle

st.write(""" 

## My First Web Application 
Let's enjoy **data science** project! 

""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')

# -- Define function to display widgets and store data


def get_input():
    # Display widgets and store their values in variables
    v_AcademicYear = st.sidebar.radio('Year', ['2562', '2563'])
    v_TCAS = st.sidebar.radio('TCAS Round', ['1', '2', '5'])
    v_GPAX = st.sidebar.slider('GPAX', 0.00, 4.00, 0.10)
    v_Sex = st.sidebar.radio('Radio', ['Male', 'Female'])
    v_SchoolRegionNameEng = st.sidebar.selectbox('Select', ['Foreign', 'Northern', 'Northeast', 'Southern', 'Central',
       'Eastern', 'Western'])
    v_Q1 = st.sidebar.radio('Q1', ['Yes', 'No'])
    v_Q2 = st.sidebar.radio('Q2', ['Yes', 'No'])
    v_Q3 = st.sidebar.radio('Q3', ['Yes', 'No'])
    v_Q4 = st.sidebar.radio('Q4', ['Yes', 'No'])
    v_Q5 = st.sidebar.radio('Q5', ['Yes', 'No'])
    v_Q6 = st.sidebar.radio('Q6', ['Yes', 'No'])
    v_Q23 = st.sidebar.radio('Q23', ['Yes', 'No'])
    v_Q24 = st.sidebar.radio('Q24', ['Yes', 'No'])
    v_Q25 = st.sidebar.radio('Q25', ['Yes', 'No'])
    v_Q26 = st.sidebar.radio('Q26', ['Yes', 'No'])
    v_Q27 = st.sidebar.radio('Q27', ['Yes', 'No'])
    v_Q28 = st.sidebar.radio('Q28', ['Yes', 'No'])
    v_Q29 = st.sidebar.radio('Q29', ['Yes', 'No'])
    v_Q30 = st.sidebar.radio('Q30', ['Yes', 'No'])
    v_Q31 = st.sidebar.radio('Q31', ['Yes', 'No'])
    v_Q32 = st.sidebar.radio('Q32', ['Yes', 'No'])
    v_Q33 = st.sidebar.radio('Q33', ['Yes', 'No'])
    v_Q34 = st.sidebar.radio('Q34', ['Yes', 'No'])
    v_Q35 = st.sidebar.radio('Q35', ['Yes', 'No'])
    v_Q36 = st.sidebar.radio('Q36', ['Yes', 'No'])
    v_Q37 = st.sidebar.radio('Q37', ['Yes', 'No'])
    v_Q38 = st.sidebar.radio('Q38', ['Yes', 'No'])
    v_Q39 = st.sidebar.radio('Q39', ['Yes', 'No'])
    v_Q40 = st.sidebar.radio('Q40', ['Yes', 'No'])
    v_Q41 = st.sidebar.radio('Q41', ['Yes', 'No'])
    v_Q42 = st.sidebar.radio('Q42', ['Yes', 'No'])
    
    # Q1
    if v_Q1 == 'Yes':
        v_Q1 = '1'
    else:
        v_Q1 = '0'
    # Q2
    if v_Q2 == 'Yes':
        v_Q2 = '1'
    else:
        v_Q2 = '0'
    # Q3
    if v_Q3 == 'Yes':
        v_Q3 = '1'
    else:
        v_Q3 = '0'
    # Q4
    if v_Q4 == 'Yes':
        v_Q4 = '1'
    else:
        v_Q4 = '0'
    # Q5
    if v_Q5 == 'Yes':
        v_Q5 = '1'
    else:
        v_Q5 = '0'
    # Q6
    if v_Q6 == 'Yes':
        v_Q6 = '1'
    else:
        v_Q6 = '0'
    # Q23
    if v_Q23 == 'Yes':
        v_Q23 = '1'
    else:
        v_Q23 = '0'
    # Q24
    if v_Q24 == 'Yes':
        v_Q24 = '1'
    else:
        v_Q24 = '0'
    # Q25
    if v_Q25 == 'Yes':
        v_Q25 = '1'
    else:
        v_Q25 = '0'
    # Q26
    if v_Q26 == 'Yes':
        v_Q26 = '1'
    else:
        v_Q26 = '0'
    # Q27
    if v_Q27 == 'Yes':
        v_Q27 = '1'
    else:
        v_Q27 = '0'
    # Q28
    if v_Q28 == 'Yes':
        v_Q28 = '1'
    else:
        v_Q28 = '0'
    # Q29
    if v_Q29 == 'Yes':
        v_Q29 = '1'
    else:
        v_Q29 = '0'
    # Q30
    if v_Q30 == 'Yes':
        v_Q30 = '1'
    else:
        v_Q30 = '0'
    # Q31
    if v_Q31 == 'Yes':
        v_Q31 = '1'
    else:
        v_Q31 = '0'
    # Q32
    if v_Q32 == 'Yes':
        v_Q32 = '1'
    else:
        v_Q32 = '0'
    # Q33
    if v_Q33 == 'Yes':
        v_Q33 = '1'
    else:
        v_Q33 = '0'
    # Q34
    if v_Q34 == 'Yes':
        v_Q34 = '1'
    else:
        v_Q34 = '0'
    # Q35
    if v_Q35 == 'Yes':
        v_Q35 = '1'
    else:
        v_Q35 = '0'
    # Q36
    if v_Q36 == 'Yes':
        v_Q36 = '1'
    else:
        v_Q36 = '0'
    # Q37
    if v_Q37 == 'Yes':
        v_Q37 = '1'
    else:
        v_Q37 = '0'
    # Q38
    if v_Q38 == 'Yes':
        v_Q38 = '1'
    else:
        v_Q38 = '0'
    # Q39
    if v_Q39 == 'Yes':
        v_Q39 = '1'
    else:
        v_Q39 = '0'
    # Q40
    if v_Q40 == 'Yes':
        v_Q40 = '1'
    else:
        v_Q40 = '0'
    # Q41
    if v_Q41 == 'Yes':
        v_Q41 = '1'
    else:
        v_Q41 = '0'
    # Q42
    if v_Q42 == 'Yes':
        v_Q42 = '1'
    else:
        v_Q42 = '0'

    # Store user input data in a dictionary
    data = {'AcademicYear': v_AcademicYear,
            'TCAS': v_TCAS,
            'GPAX': v_GPAX,
            'Sex': v_Sex,
            'SchoolRegionNameEng': v_SchoolRegionNameEng,
            'Q1': v_Q1,
            'Q2': v_Q2,
            'Q3': v_Q3,
            'Q4': v_Q4,
            'Q5': v_Q5,
            'Q6': v_Q6,
            'Q23': v_Q23,
            'Q24': v_Q24,
            'Q25': v_Q25,
            'Q26': v_Q26,
            'Q27': v_Q27,
            'Q28': v_Q28,
            'Q29': v_Q29,
            'Q30': v_Q30,
            'Q31': v_Q31,
            'Q32': v_Q32,
            'Q33': v_Q33,
            'Q34': v_Q34,
            'Q35': v_Q35,
            'Q36': v_Q36,
            'Q37': v_Q37,
            'Q38': v_Q38,
            'Q39': v_Q39,
            'Q40': v_Q40,
            'Q41': v_Q41,
            'Q42': v_Q42}

    # Create a data frame from the above dictionary
    data_df = pd.DataFrame(data, index=[0])
    return data_df

# -- Call function to display widgets and get data from user
df = get_input()

st.header('Application of Status Prediction:')

# -- Display new data from user inputs:
st.subheader('User Input:')
st.write(df)

# -- Data Pre-processing for New Data:
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('newdf2.csv')
df = pd.concat([df, data_sample],axis=0)

###Data Cleaning & Feature Engineering###

#One-hot encoding for nominal features
cat_data = pd.get_dummies(df[['Sex','SchoolRegionNameEng']])

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)

#drop
X_new = X_new.drop(columns=['Unnamed: 0'])
X_new = X_new.drop(columns=['Sex'])
X_new = X_new.drop(columns=['SchoolRegionNameEng'])

# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(X_new)

# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)

st.subheader('Normalization Input:')
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)

st.subheader('Prediction:')
st.write(prediction)
