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


load_dotenv()
URLBDD = os.getenv("URLBDD")

model_trained = joblib.load("model_trained")

app = FastAPI()

class Features(BaseModel):
    # === TRÈS IMPORTANT (top 3) ===
    ext_source_2: float = Field(
        ..., description="Score externe normalisé (0 à 1) d'un organisme tiers évaluant la fiabilité du client. Plus proche de 1 = meilleur profil."
    )
    ext_source_3: float = Field(
        ..., description="Second score externe normalisé (0 à 1), similaire à ext_source_2 mais d'une autre source."
    )
    days_birth: int = Field(
        ..., description="Âge du client en nombre de jours, en négatif (ex: -9461 jours ≈ 26 ans). Diviser par -365 pour obtenir l'âge en années."
    )
 
    # === IMPORTANT ===
    amt_annuity: float = Field(
        ..., description="Montant de l'annuité (mensualité) du crédit demandé."
    )
    amt_payment: float = Field(
        ..., description="Montant réellement payé lors de la dernière échéance de remboursement."
    )
    amt_credit_y: float = Field(
        ..., description="Montant total du crédit accordé pour la demande actuelle."
    )
    cnt_payment: int = Field(
        ..., description="Nombre d'échéances de paiement prévues pour un crédit précédent."
    )
    amt_goods_price: float = Field(
        ..., description="Prix du bien pour lequel le crédit est demandé (ex: prix de la voiture, du bien immobilier)."
    )
    days_entry_payment: int = Field(
        ..., description="Nombre de jours avant aujourd'hui où le dernier paiement a été effectivement réalisé (négatif)."
    )
    days_employed: int = Field(
        ..., description="Ancienneté professionnelle en nombre de jours, en négatif (ex: -637 jours ≈ 1.7 an dans l'emploi actuel)."
    )
 
    # === MOYENNEMENT IMPORTANT ===
    amt_credit_sum: float = Field(
        ..., description="Montant total des crédits en cours déclarés auprès d'autres organismes (bureau de crédit)."
    )
    days_instalment: int = Field(
        ..., description="Nombre de jours avant aujourd'hui où l'échéance de paiement était théoriquement prévue (négatif)."
    )
    num_instalment_number: int = Field(
        ..., description="Numéro de l'échéance dans le plan de remboursement (ex: 5 = 5e mensualité payée)."
    )
    amt_instalment: float = Field(
        ..., description="Montant attendu pour l'échéance de remboursement (avant paiement réel)."
    )
    days_credit_enddate: int = Field(
        ..., description="Nombre de jours restants avant la fin prévue d'un crédit en cours (négatif si déjà censé être terminé)."
    )
    days_credit: int = Field(
        ..., description="Nombre de jours écoulés depuis l'ouverture d'un crédit déclaré au bureau de crédit (négatif)."
    )
    amt_credit_sum_debt: float = Field(
        ..., description="Montant restant dû sur les crédits en cours déclarés au bureau de crédit."
    )
    days_decision: int = Field(
        ..., description="Nombre de jours avant aujourd'hui où la décision sur une demande de crédit précédente a été prise (négatif)."
    )
    amt_application: float = Field(
        ..., description="Montant demandé par le client lors d'une précédente demande de crédit."
    )
    days_credit_update: int = Field(
        ..., description="Nombre de jours depuis la dernière mise à jour des informations d'un crédit au bureau de crédit (négatif)."
    )
    days_enddate_fact: int = Field(
        ..., description="Nombre de jours avant aujourd'hui où un crédit a été effectivement clôturé (négatif)."
    )
    amt_credit_x: float = Field(
        ..., description="Montant du crédit accordé lors d'une précédente demande."
    )
    amt_income_total: float = Field(
        ..., description="Revenu total annuel déclaré par le client."
    )
    amt_credit_max_overdue: float = Field(
        ..., description="Montant maximum jamais resté impayé (en retard) sur un crédit déclaré au bureau de crédit."
    )
 
    # === MOINS IMPORTANT (impact plus faible, mais gardé pour cohérence avec le modèle) ===
    region_rating_client: int = Field(
        ..., description="Note de la région de résidence du client (1 = meilleure, 3 = moins bonne)."
    )
    cnt_children: int = Field(..., description="Nombre d'enfants à charge.")
    cnt_fam_members: int = Field(..., description="Nombre total de personnes dans le foyer.")
 
    genre: Literal['M', 'F', 'XNA'] = Field(..., description="Genre du client.")
    possede_voiture: Literal['Y', 'N'] = Field(..., description="Le client possède-t-il une voiture ?")
    possede_logement: Literal['Y', 'N'] = Field(..., description="Le client est-il propriétaire de son logement ?")
    type_revenu: Literal[
        'Working', 'State servant', 'Commercial associate', 'Pensioner',
        'Unemployed', 'Student', 'Businessman', 'Maternity leave'
    ] = Field(..., description="Type de source de revenu du client (salarié, fonctionnaire, retraité, etc.).")
    niveau_education: Literal[
        'Secondary / secondary special', 'Higher education', 'Incomplete higher',
        'Lower secondary', 'Academic degree'
    ] = Field(..., description="Niveau d'études le plus élevé atteint.")
    statut_marital: Literal[
        'Single / not married', 'Married', 'Civil marriage', 'Widow',
        'Separated', 'Unknown'
    ] = Field(..., description="Statut marital du client.")
    type_logement: Literal[
        'House / apartment', 'Rented apartment', 'With parents',
        'Municipal apartment', 'Office apartment', 'Co-op apartment'
    ] = Field(..., description="Type de logement occupé.")
    profession: Literal[
        'Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers',
        'Sales staff', 'Cleaning staff', 'Cooking staff', 'Private service staff',
        'Medicine staff', 'Security staff', 'High skill tech staff',
        'Waiters/barmen staff', 'Low-skill Laborers', 'Realty agents',
        'Secretaries', 'IT staff', 'HR staff'
    ] = Field(..., description="Catégorie professionnelle du client.")
 
    # === PRÉRÉGLAGES (regroupent plusieurs colonnes peu importantes chacune) ===
    profil_credits_bureau: Literal['Aucun', 'Peu de crédits', 'Profil moyen', 'Fort utilisateur de crédit'] = Field(
        ..., description="Résume les types de crédits déclarés auprès d'autres organismes (conso, carte, immo...)."
    )
    profil_statut_bureau: Literal['Aucun historique', 'Peu de crédits actifs/clôturés', 'Historique moyen', 'Historique chargé'] = Field(
        ..., description="Résume le nombre de crédits actifs/clôturés au bureau de crédit."
    )
    profil_demandes_precedentes: Literal['Aucune demande', 'Peu de demandes, bon profil', 'Demandes mixtes (refus/annulations)', 'Nombreuses demandes'] = Field(
        ..., description="Résume l'historique des demandes de crédit précédentes (approuvées, refusées, annulées)."
    )
    profil_type_contrat_precedent: Literal['Aucun', 'Principalement crédit conso', 'Principalement prêt cash', 'Mixte'] = Field(
        ..., description="Résume le type de contrats de crédit précédemment souscrits."
    )
    profil_motif_decision_precedente: Literal['Aucun', 'Approbations standards (XAP)', 'Historique de refus (HC/LIMIT/SCO)', 'Mixte'] = Field(
        ..., description="Résume le motif des décisions prises sur les demandes de crédit précédentes."
    )

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