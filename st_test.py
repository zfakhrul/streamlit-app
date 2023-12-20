import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Bank loan approval predictor")
st.text("This is the web app thats applies an ML model in predicting bank loan approval")

gender = st.radio("Gender", ["Female", "Male"])

married = st.radio("Are you married?", ["No", "Yes"])

dependents = st.slider("Dependents", 0, 10)

education = st.radio("Education",["Graduate", "Not Graduate"])

selfEmp = st.radio("Self Employed?", ["No", "Yes"])

appIncome = st.text_input("Applicant Income")
coIncome = st.text_input("Co-Applicant Income")
LoanAmt = st.text_input("Loan Amount")
term = st.slider("Terms",0,360)
CreditHistory = 1
area = st.selectbox("Select Area",["Semi-Urban","Urban"])

#import pandas as pd
#data = pd.read_csv("dataset/Dataset3-Telco-Customer-Churn.csv")
#data

click = st.button("Submit")

if click:
    if gender=="Female":
        gender = 0
    else: 
        gender = 1
        
    if married=="No":
        married = 0
    else:
        married = 1
    
    if education =="Graduate":
        education = 0
    else:
        education = 1
    
    if selfEmp == "No":
        selfEmp = 0
    else:
        selfEmp = 1
        
    if area == "Semi-Urban":
        area = 0
    else:
        area = 1
        
        
    scaler = pickle.load(open('scaler.pkl','rb'))
        
    dat = [[gender, married, dependents, education, selfEmp, appIncome, coIncome, LoanAmt, term, CreditHistory, area]]

    dat = scaler.transform(dat)
    
    st.text(dat)
    
    model = pickle.load(open('svm_model.pkl','rb'))
    res = model.predict(dat)

    if res==1:
        st.text("Your application will most likely to be approved! :)")
    else:
        st.text("Your application will most probably be rejected! :(")
