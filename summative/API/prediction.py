# Importing the required libraries
from pathlib import Path
from io import BytesIO
from threading import Lock
import os
import secrets
from fastapi import Header, Depends

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File

from pydantic import BaseModel, Field, ConfigDict
from sklearn.base import clone
# Importing CORS middleware to control cross-origin API requests
from fastapi.middleware.cors import CORSMiddleware


# Creating the FastAPI application
app = FastAPI(
    title="South African Commuting Time Prediction API",
    description="Predicts commuting duration using the 2020 National Household Travel Survey.",
    version="1.0.0"
)
# Configuring the origins allowed to access the API from browsers
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]

# Adding CORS middleware to our FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

# Reading the retraining API key from environment variables
RETRAIN_API_KEY = os.getenv("RETRAIN_API_KEY")


# Verifying authorization before allowing model retraining
def verify_retrain_key(x_api_key: str = Header(...)):

    # Rejecting retraining if the server has no configured API key
    if not RETRAIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Retraining is not configured."
        )

    # Comparing the provided key securely
    if not secrets.compare_digest(x_api_key, RETRAIN_API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid retraining API key."
        )
# Defining paths relative to this Python file
API_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = API_DIRECTORY / "best_model.pkl"
TRAINING_DATA_PATH = API_DIRECTORY / "training_data.csv"

# Defining the target and expected feature order
TARGET_COLUMN = "Tot_time_travel_work"

FEATURE_COLUMNS = [
    "Age",
    "Q49Days",
    "WorkStartMinutes",
    "Sex",
    "Q43SectOfWork",
    "Q48bDMunic",
    "Q413EmployCash",
    "Mainmode_work"
]

REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Limiting uploaded CSV files to 5 MB
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# Protecting model updates from simultaneous requests
model_lock = Lock()

# Loading the saved model and original training data
model = joblib.load(MODEL_PATH)
training_data = pd.read_csv(TRAINING_DATA_PATH)


# Defining validation rules for commuter information
class CommuterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Age: int = Field(ge=15, le=120)
    Sex: int = Field(ge=1, le=2)
    Q43SectOfWork: int = Field(ge=1, le=3)
    Q48bDMunic: int = Field(ge=11, le=70)
    Q49Days: int = Field(ge=1, le=7)
    WorkStartMinutes: int = Field(ge=0, le=1439)
    Q413EmployCash: int = Field(ge=1, le=2)
    Mainmode_work: int = Field(ge=1, le=7)


# Creating a health-check endpoint
@app.get("/")
def home():
    return {
        "message": "Commuting Time Prediction API is running",
        "status": "success"
    }


# Creating the prediction endpoint
@app.post("/predict")
def predict_commuting_time(commuter: CommuterInput):

    # Converting validated inputs into a DataFrame
    input_data = pd.DataFrame(
        [commuter.model_dump()]
    )[FEATURE_COLUMNS]

    try:
        # Preventing model replacement during prediction
        with model_lock:
            prediction = model.predict(input_data)[0]

        return {
            "predicted_commuting_time": round(float(prediction), 2),
            "unit": "minutes"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate a commuting time prediction."
        ) from error


# Creating the retraining endpoint
@app.post("/retrain")
async def retrain_model(
    file: UploadFile = File(...),
    _: None = Depends(verify_retrain_key)
):
    global model, training_data

    # Accepting CSV uploads only
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    # Reading the uploaded file with a size limit
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="CSV file exceeds the 5 MB limit."
        )

    # Parsing the CSV file
    try:
        new_data = pd.read_csv(BytesIO(contents))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError):
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded CSV file."
        )

    # Checking that all required columns exist
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in new_data.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {missing_columns}"
        )

    # Rejecting empty datasets
    if new_data.empty:
        raise HTTPException(
            status_code=422,
            detail="The uploaded CSV contains no records."
        )

    # Rejecting unexpected columns
    unexpected_columns = [
        column for column in new_data.columns
        if column not in REQUIRED_COLUMNS
    ]

    if unexpected_columns:
        raise HTTPException(
            status_code=422,
            detail=f"Unexpected columns: {unexpected_columns}"
        )

    # Validating each commuter record using the same prediction rules
    for row_number, row in enumerate(
        new_data[FEATURE_COLUMNS].to_dict(orient="records"),
        start=2
    ):
        try:
            CommuterInput.model_validate(row)
        except Exception as error:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid commuter data at CSV row {row_number}: {error}"
            )

    # Validating the target column
    target_values = pd.to_numeric(
        new_data[TARGET_COLUMN],
        errors="coerce"
    )

    if (
        target_values.isna().any()
        or not target_values.between(1, 365).all()
    ):
        raise HTTPException(
            status_code=422,
            detail="Commuting duration must be between 1 and 365 minutes."
        )

    new_data = new_data[REQUIRED_COLUMNS].copy()
    new_data[TARGET_COLUMN] = target_values

    try:
        with model_lock:
            # Combining existing training records with uploaded records
            combined_data = pd.concat(
                [training_data, new_data],
                ignore_index=True
            )

            # Separating features and target
            X_updated = combined_data[FEATURE_COLUMNS]
            y_updated = combined_data[TARGET_COLUMN]

            # Creating a fresh copy of our existing pipeline
            updated_model = clone(model)

            # Retraining on the combined dataset
            updated_model.fit(X_updated, y_updated)

            # Saving the updated model
            joblib.dump(updated_model, MODEL_PATH)

            # Saving the expanded training dataset
            combined_data.to_csv(TRAINING_DATA_PATH, index=False)

            # Activating the updated model without restarting the API
            model = updated_model
            training_data = combined_data

        return {
            "message": "Model retrained successfully",
            "new_records": len(new_data),
            "total_training_records": len(combined_data),
            "model_updated": True
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Model retraining failed."
        ) from error