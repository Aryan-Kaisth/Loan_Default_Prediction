from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.logger import logging
from src.exception import CustomException
from app.schemas import LoanRequest

app = FastAPI(title="Loan Default Prediction")

# Load model and preprocessor once at startup
pipeline = PredictionPipeline()

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def build_features(data: LoanRequest) -> pd.DataFrame:
    """
    Convert LoanRequest to a DataFrame for prediction.
    Includes computed fields.
    """
    features = {
        "Age": data.Age,
        "Income": data.Income,
        "LoanAmount": data.LoanAmount,
        "CreditScore": data.CreditScore,
        "MonthsEmployed": data.MonthsEmployed,
        "NumCreditLines": data.NumCreditLines,
        "InterestRate": data.InterestRate,
        "LoanTerm": data.LoanTerm,
        "DTIRatio": data.DTIRatio,
        "Education": data.Education,
        "EmploymentType": data.EmploymentType,
        "MaritalStatus": data.MaritalStatus,
        "HasMortgage": data.HasMortgage,
        "HasDependents": data.HasDependents,
        "LoanPurpose": data.LoanPurpose,
        "HasCoSigner": data.HasCoSigner,
        # Computed fields
        "LoanPerCreditScore": data.LoanPerCreditScore,
        "IncomePerCreditLine": data.IncomePerCreditLine,
        "InterestOverIncome": data.InterestOverIncome,
        "MonthlyPayment": data.MonthlyPayment,
        "CreditScorePerAge": data.CreditScorePerAge,
        "EmploymentStability": data.EmploymentStability,
        "DTI_LoanRatio": data.DTI_LoanRatio,
        "IncomePerLoanTerm": data.IncomePerLoanTerm,
        "LTI": data.LTI,
        "YearsEmployed": data.YearsEmployed,
        "EmploymentToLoanTerm": data.EmploymentToLoanTerm,
    }
    return pd.DataFrame([features])


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the HTML form for user input.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_loan(request: Request, data: LoanRequest):
    """
    Predict loan default risk from user input.
    """
    try:
        logging.info(f"Received prediction request: {data.dict()}")
        df = build_features(data)
        pred = pipeline.predict(df)[0]  # 0 or 1
        logging.info(f"Prediction result: {pred}")

        result = "Yes" if pred == 1 else "No"
        return {
            "predicted_risk": result,
            "raw_prediction": int(pred)  # optional, keep numeric value too
        }

    except CustomException as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")
