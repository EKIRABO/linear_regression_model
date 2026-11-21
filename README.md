# CommuteInsight SA

## 1. Mission and Problem

Unpredictable commuting times can lead to late arrivals, lost productivity, and difficulties planning daily routines, highlighting a broader challenge of urban mobility in South Africa. Aligned with the Grand Challenge of Urbanization, CommuteInsight SA explores how technology can help people navigate everyday transportation challenges. Using machine learning and real-world travel survey data, the project estimates workplace commuting time to support more informed travel planning and contribute to smarter, more accessible urban mobility solutions.

## 2. Dataset Description

The project uses the **South African National Household Travel Survey (NHTS) 2020**, obtained from the University of Cape Town's DataFirst portal.

The dataset contains information about individuals' travel patterns, employment, demographics, and transportation choices. After cleaning and preprocessing, **24,492 records** were retained, with **eight independent variables** used to predict total workplace commuting time in minutes.

**Dataset source:** [https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/886/get-microdata](https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/886/get-microdata)

**Access:** To download the original dataset, users must register or log in to the DataFirst portal, specify the purpose for which they need the data, and agree to the data usage terms.

## 3. Swagger UI — Public API

CommuteInsight SA uses a FastAPI backend deployed on Render to provide predictions and support model retraining.

**Swagger UI:** [https://commuting-time-api.onrender.com/docs](https://commuting-time-api.onrender.com/docs)

### Available Endpoints

- **`POST /predict`** — Accepts eight predictor values and returns an estimated workplace commuting duration in minutes.
- **`POST /retrain`** — Retrains the regression model using an uploaded CSV dataset.

### How to Test the API

**Prediction:**

1. Open Swagger UI and expand `POST /predict`.
2. Click **Try it out**.
3. Enter valid values for the eight predictors using the example request format.
4. Click **Execute** to view the predicted commuting time.

**Retraining:**

1. Prepare a CSV containing the eight predictor columns and the target variable (`Tot_time_travel_work`).
2. Expand `POST /retrain` and click **Try it out**.
3. Upload the CSV and click **Execute** to view the response.

**Note:** Retraining may update the deployed model. Use appropriately prepared data when testing this endpoint.

## 4. YouTube Video Demonstration

**Video link:** To be added after recording and uploading the demonstration.

## 5. How to Run the Mobile Application

### Prerequisites

- Flutter SDK and Dart
- Android Studio or Visual Studio Code
- Android emulator or physical Android device
- Internet connection

### Installation and Setup

**Step 1: Clone the repository**

```bash
git clone https://github.com/EKIRABO/linear_regression_model.git
```

**Step 2: Navigate to the Flutter application**

```bash
cd linear_regression_model/summative/FlutterApp
```

**Step 3: Install Flutter dependencies**

```bash
flutter pub get
```

**Step 4: Start an Android emulator or connect a physical Android device.**

**Step 5: Run the application**

```bash
flutter run
```

### Using the Application

1. Open CommuteInsight SA on your device.
2. Enter the required demographic, employment, and commuting information.
3. Select the relevant transport mode and work start time.
4. Tap **Predict** to receive the estimated workplace commuting duration.

The application communicates with the deployed FastAPI backend to generate predictions.

---

### Project Structure

```text
linear_regression_model/
├── README.md
└── summative/
    ├── pyproject.toml
    ├── uv.lock
    ├── linear_regression/
    │   └── multivariate.ipynb
    ├── API/
    │   ├── prediction.py
    │   ├── requirements.txt
    │   ├── best_model.pkl
    │   └── training_data.csv
    └── FlutterApp/
        ├── lib/
        ├── android/
        └── pubspec.yaml
```