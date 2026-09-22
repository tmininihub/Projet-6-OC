from fastapi.testclient import TestClient
from main import app

FEATURES_VALIDES = {
    "Active_x": 2, "Bad_debt": 0, "Closed": 6, "Sold": 0,
    "Another_type_of_loan": 0, "Car_loan": 0, "Cash_loan_non_earmarked": 0,
    "Consumer_credit": 4, "Credit_card": 4, "Interbank_credit": 0,
    "Loan_for_business_development": 0, "Loan_for_purchase_of_shares_margin_lending": 0,
    "Loan_for_the_purchase_of_equipment": 0, "Loan_for_working_capital_replenishment": 0,
    "Microloan": 0, "Mobile_operator_loan": 0, "Mortgage": 0,
    "Real_estate_loan": 0, "Unknown_type_of_loan": 0,
    "DAYS_CREDIT": -874, "CREDIT_DAY_OVERDUE": 0, "DAYS_CREDIT_ENDDATE": -349,
    "DAYS_ENDDATE_FACT": -697, "AMT_CREDIT_MAX_OVERDUE": 1681, "CNT_CREDIT_PROLONG": 0,
    "AMT_CREDIT_SUM": 108131, "AMT_CREDIT_SUM_DEBT": 49156, "AMT_CREDIT_SUM_LIMIT": 7997,
    "DAYS_CREDIT_UPDATE": -499,
    "Approved_x": 1, "Canceled": 0, "Refused_x": 0, "Unused_offer": 0,
    "Cash_loans": 0, "Consumer_loans": 1, "Revolving_loans": 0, "XNA_x": 0,
    "CLIENT": 0, "HC": 0, "LIMIT": 0, "SCO": 0, "SCOFR": 0, "SYSTEM": 0,
    "VERIF": 0, "XAP": 1, "XNA_y": 0,
    "AMT_CREDIT_x": 179055, "AMT_APPLICATION": 179055, "DAYS_DECISION": -606,
    "CNT_PAYMENT": 24, "NUM_INSTALMENT_NUMBER": 190, "DAYS_INSTALMENT": -5605,
    "DAYS_ENTRY_PAYMENT": -315, "AMT_INSTALMENT": 11559, "AMT_PAYMENT": 11559,
    "M": 1, "F": 0, "XNA": 0, "N_x": 1, "Y_x": 0, "Y_y": 1, "N_y": 0,
    "CNT_CHILDREN": 0, "AMT_INCOME_TOTAL": 202500, "AMT_CREDIT_y": 406597,
    "AMT_ANNUITY": 24700, "AMT_GOODS_PRICE": 351000,
    "Working": 1, "State_servant": 0, "Commercial_associate": 0, "Pensioner": 0,
    "Unemployed": 0, "Student": 0, "Businessman": 0, "Maternity_leave": 0,
    "Secondary_secondary_special": 1, "Higher_education": 0, "Incomplete_higher": 0,
    "Lower_secondary": 0, "Academic_degree": 0,
    "Single_not_married": 1, "Married": 0, "Civil_marriage": 0, "Widow": 0,
    "Separated": 0, "Unknown": 0,
    "House_apartment": 1, "Rented_apartment": 0, "With_parents": 0,
    "Municipal_apartment": 0, "Office_apartment": 0, "Co_op_apartment": 0,
    "DAYS_BIRTH": -9461, "DAYS_EMPLOYED": -637,
    "Laborers": 1, "Core_staff": 0, "Accountants": 0, "Managers": 0, "Drivers": 0,
    "Sales_staff": 0, "Cleaning_staff": 0, "Cooking_staff": 0,
    "Private_service_staff": 0, "Medicine_staff": 0, "Security_staff": 0,
    "High_skill_tech_staff": 0, "Waiters_barmen_staff": 0, "Low_skill_Laborers": 0,
    "Realty_agents": 0, "Secretaries": 0, "IT_staff": 0, "HR_staff": 0,
    "CNT_FAM_MEMBERS": 1, "REGION_RATING_CLIENT": 2,
    "EXT_SOURCE_2": 0.2629485927471776, "EXT_SOURCE_3": 0.1393757800997895,
}

client = TestClient(app)

def test_flux():
    request = client.post("/Request", json=FEATURES_VALIDES)
    assert request.status_code == 200
    request = request.json()
    id = request
    assert len(id) == 36

    predict = client.post("/Predict", params={"id" : id})
    assert predict.status_code == 200
    predict = predict.json()
    credit_accept = predict[1]
    credit_decline = predict[2]
    assert credit_accept + credit_decline == 100








