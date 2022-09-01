
from pydantic import BaseModel, Field
import joblib
import pickle
import re
from fastapi import FastAPI
from lightgbm import LGBMClassifier
import pandas as pd
import os
import uvicorn
from classes import ChargeOff
    
app = FastAPI(debug=True, title = 'Loan Risks Prediction API', version = 1.0, description = 'Simple API to predict Loan risks.')

#creating the classifier
pickle_in = open('models/lgbmodel.pkl', 'rb')
model = pickle.load(pickle_in)
pickle_in.close()

THRESHOLD = 0.5

@app.get('/')
def get_root():
    return {'message': 'Welcome to the loan risk detection api v4'}

'''@app.post('/chargeoff_detection/')
def predict_chargeoff(data:ChargeOff):

    inp = [data.NAME_CONTRACT_TYPE, data.CODE_GENDER, data.FLAG_OWN_CAR, data.FLAG_OWN_REALTY,
       data.CNT_CHILDREN, data.AMT_INCOME_TOTAL, data.AMT_CREDIT, data.NAME_TYPE_SUITE,
       data.NAME_INCOME_TYPE, data.NAME_EDUCATION_TYPE, data.NAME_FAMILY_STATUS,
       data.NAME_HOUSING_TYPE, data.REGION_POPULATION_RELATIVE, data.DAYS_BIRTH,
       data.DAYS_EMPLOYED, data.DAYS_REGISTRATION, data.DAYS_ID_PUBLISH, data.OWN_CAR_AGE,
       data.FLAG_EMP_PHONE, data.FLAG_WORK_PHONE, data.FLAG_PHONE, data.FLAG_EMAIL,
       data.OCCUPATION_TYPE, data.REGION_RATING_CLIENT, data.WEEKDAY_APPR_PROCESS_START,
       data.HOUR_APPR_PROCESS_START, data.ORGANIZATION_TYPE, data.APARTMENTS_AVG,
       data.BASEMENTAREA_AVG, data.YEARS_BEGINEXPLUATATION_AVG, data.YEARS_BUILD_AVG,
       data.COMMONAREA_AVG, data.ELEVATORS_AVG, data.ENTRANCES_AVG, data.FLOORSMAX_AVG,
       data.FLOORSMIN_AVG, data.LANDAREA_AVG, data.LIVINGAPARTMENTS_AVG,
       data.LIVINGAREA_AVG, data.NONLIVINGAPARTMENTS_AVG, data.NONLIVINGAREA_AVG,
       data.OBS_60_CNT_SOCIAL_CIRCLE, data.DEF_60_CNT_SOCIAL_CIRCLE,
       data.DAYS_LAST_PHONE_CHANGE, data.FLAG_DOCUMENT_3, data.FLAG_DOCUMENT_5,
       data.FLAG_DOCUMENT_6, data.FLAG_DOCUMENT_8, data.FLAG_DOCUMENT_9,
       data.FLAG_DOCUMENT_11, data.FLAG_DOCUMENT_13, data.FLAG_DOCUMENT_16,
       data.FLAG_DOCUMENT_18, data.AMT_REQ_CREDIT_BUREAU_HOUR,
       data.AMT_REQ_CREDIT_BUREAU_DAY, data.AMT_REQ_CREDIT_BUREAU_WEEK,
       data.AMT_REQ_CREDIT_BUREAU_MON, data.AMT_REQ_CREDIT_BUREAU_QRT,
       data.AMT_REQ_CREDIT_BUREAU_YEAR, data.ADDRESS_MISMATCH, data.EXT_SOURCE_AVG,
       data.CREDITS_ACTIVE, data.CREDITS_CLOSED, data.ACTIVE_DEBT_SUM, data.PREV_APPROVED,
       data.PREV_CANCELED, data.PREV_REFUSED, data.INSTALLMENTS_LEFT, data.AMT_BALANCE,
       data.AMT_CREDIT_LIMIT_ACTUAL, data.AMT_DRAWINGS_ATM_CURRENT,
       data.AMT_DRAWINGS_CURRENT, data.AMT_DRAWINGS_OTHER_CURRENT,
       data.AMT_DRAWINGS_POS_CURRENT, data.AMT_INST_MIN_REGULARITY,
       data.AMT_PAYMENT_CURRENT, data.AMT_PAYMENT_TOTAL_CURRENT,
       data.AMT_RECEIVABLE_PRINCIPAL, data.AMT_RECIVABLE, data.AMT_TOTAL_RECEIVABLE]

    prob = model.predict_proba([inp])[0][1]
    prediction = int(prob > THRESHOLD)
    return {'prediction':prediction, 'probability':prob}
'''

@app.post('/chargeoff_detection/')
def predict_chargeoff(data:ChargeOff):

    columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'NAME_TYPE_SUITE',
       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
       'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
       'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'OCCUPATION_TYPE', 'REGION_RATING_CLIENT', 'WEEKDAY_APPR_PROCESS_START',
       'HOUR_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'APARTMENTS_AVG',
       'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
       'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG',
       'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
       'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_16',
       'FLAG_DOCUMENT_18', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR', 'ADDRESS_MISMATCH', 'EXT_SOURCE_AVG',
       'CREDITS_ACTIVE', 'CREDITS_CLOSED', 'ACTIVE_DEBT_SUM', 'PREV_APPROVED',
       'PREV_CANCELED', 'PREV_REFUSED', 'INSTALLMENTS_LEFT', 'AMT_BALANCE',
       'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
       'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
       'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
       'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
       'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE']
    
    data = data.dict()
    inp = [value for (key,value) in data.items()]
    df = pd.DataFrame([inp], columns=columns)
    prob = model.predict_proba(df)[0][1].tolist()
    prediction = int(prob > THRESHOLD)
    return {'prediction':prediction, 'probability':prob}


#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8000)
