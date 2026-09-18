import fastapi
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import uvicorn
import pandas as pd
import joblib
import dotenv
from dotenv import load_dotenv
import os
import sqlalchemy
from sqlalchemy import create_engine, engine, text
import uuid

if os.path.exists("/.dockerenv"):
    load_dotenv("docker.env")
else:
    load_dotenv("local.env")

URLBDD = os.getenv("URLBDD")

model_trained = joblib.load("model_trained")

app = FastAPI()

class Features(BaseModel):
    Active_x: int
    Bad_debt: int
    Closed: int
    Sold: int
    Another_type_of_loan: int
    Car_loan: int
    Cash_loan_non_earmarked: int
    Consumer_credit: int
    Credit_card: int
    Interbank_credit: int
    Loan_for_business_development: int
    Loan_for_purchase_of_shares_margin_lending: int
    Loan_for_the_purchase_of_equipment: int
    Loan_for_working_capital_replenishment: int
    Microloan: int
    Mobile_operator_loan: int
    Mortgage: int
    Real_estate_loan: int
    Unknown_type_of_loan: int
    DAYS_CREDIT: int
    CREDIT_DAY_OVERDUE: int
    DAYS_CREDIT_ENDDATE: int
    DAYS_ENDDATE_FACT: int
    AMT_CREDIT_MAX_OVERDUE: int
    CNT_CREDIT_PROLONG: int
    AMT_CREDIT_SUM: int
    AMT_CREDIT_SUM_DEBT: int
    AMT_CREDIT_SUM_LIMIT: int
    DAYS_CREDIT_UPDATE: int
    Approved_x: int
    Canceled: int
    Refused_x: int
    Unused_offer: int
    Cash_loans: int
    Consumer_loans: int
    Revolving_loans: int
    XNA_x: int
    CLIENT: int
    HC: int
    LIMIT: int
    SCO: int
    SCOFR: int
    SYSTEM: int
    VERIF: int
    XAP: int
    XNA_y: int
    AMT_CREDIT_x: int
    AMT_APPLICATION: int
    DAYS_DECISION: int
    CNT_PAYMENT: int
    NUM_INSTALMENT_NUMBER: int
    DAYS_INSTALMENT: int
    DAYS_ENTRY_PAYMENT: int
    AMT_INSTALMENT: int
    AMT_PAYMENT: int
    M: int
    F: int
    XNA: int
    N_x: int
    Y_x: int
    Y_y: int
    N_y: int
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: int
    AMT_CREDIT_y: int
    AMT_ANNUITY: int
    AMT_GOODS_PRICE: int
    Working: int
    State_servant: int
    Commercial_associate: int
    Pensioner: int
    Unemployed: int
    Student: int
    Businessman: int
    Maternity_leave: int
    Secondary_secondary_special: int
    Higher_education: int
    Incomplete_higher: int
    Lower_secondary: int
    Academic_degree: int
    Single_not_married: int
    Married: int
    Civil_marriage: int
    Widow: int
    Separated: int
    Unknown: int
    House_apartment: int
    Rented_apartment: int
    With_parents: int
    Municipal_apartment: int
    Office_apartment: int
    Co_op_apartment: int
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    Laborers: int
    Core_staff: int
    Accountants: int
    Managers: int
    Drivers: int
    Sales_staff: int
    Cleaning_staff: int
    Cooking_staff: int
    Private_service_staff: int
    Medicine_staff: int
    Security_staff: int
    High_skill_tech_staff: int
    Waiters_barmen_staff: int
    Low_skill_Laborers: int
    Realty_agents: int
    Secretaries: int
    IT_staff: int
    HR_staff: int
    CNT_FAM_MEMBERS: int
    REGION_RATING_CLIENT: int
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float

from fastapi.responses import FileResponse

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/Request")
def Request(features :Features):
    features = features.model_dump()
    features["id"] = str(uuid.uuid4())
    df_features = pd.DataFrame([features])

    engine = create_engine(URLBDD)
    df_features.to_sql(
        "Project6_Features",
        con=engine,
        if_exists="append",
        index=False
    )
    print(features["id"])
    return features["id"]

@app.post("/Predict")
def Predict(id: str):
    engine = create_engine(URLBDD)

    with engine.connect() as conn:
        result = conn.execute(
            text('SELECT * FROM "Project6_Features" WHERE id = :id'),
            {"id": id}
        )
        row = result.fetchone()
    df_row = pd.DataFrame([dict(row._mapping)])
    df_row = df_row.drop(columns="id")
    y_test_pred = model_trained.predict_proba(df_row)
    for i in y_test_pred:
        output = i
    print(f"Crédit Accordé : {output[0]*100:.2f}%, Crédit Refusé : {output[1]*100:.2f}%")

    return f"Crédit Accordé : {output[0]*100:.2f}%, Crédit Refusé : {output[1]*100:.2f}%"