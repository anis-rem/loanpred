from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import traceback

app = Flask(__name__)
CORS(app)

# Global variables for model and encoder
model = None
label_encoders = {}
model_features = None


def validate_model(loaded_model, test_features=None):
    """Validate that the loaded object is actually a scikit-learn model"""
    try:
        # Check if it has the required methods
        if not hasattr(loaded_model, 'predict'):
            print(f"❌ Loaded object doesn't have 'predict' method. Type: {type(loaded_model)}")
            return False

        if not hasattr(loaded_model, 'predict_proba'):
            print(f"❌ Loaded object doesn't have 'predict_proba' method. Type: {type(loaded_model)}")
            return False

        # Use provided test features or create dummy data
        if test_features is not None:
            dummy_features = test_features
        else:
            # Create dummy features matching the expected 15 features
            dummy_features = np.array([[25, 0, 3, 50000, 2, 3, 10000, 0, 10.5, 0.2, 5, 600, 0, 0.2, 2000]])

        try:
            pred = loaded_model.predict(dummy_features)
            pred_proba = loaded_model.predict_proba(dummy_features)
            print(f"✅ Model validation successful. Prediction shape: {pred.shape}, Proba shape: {pred_proba.shape}")
            return True
        except Exception as e:
            print(f"❌ Model validation failed during prediction test: {e}")
            return False

    except Exception as e:
        print(f"❌ Model validation error: {e}")
        return False


def load_model_safe():
    """Safe model loading function with fallback options"""
    global model, model_features, label_encoders

    script_dir = Path(__file__).parent.absolute()
    print(f"\n🔍 Searching for model files in: {script_dir}")

    # Possible model file paths (in order of preference)
    model_files = [
        script_dir / 'loan_model_recreated.pkl',
        script_dir / 'BESTLOANPRED.pkl',
        script_dir / 'loan_model.pkl',
        script_dir / 'loan_model_backup.pkl'
    ]

    # Try to load encoders first
    encoders_path = script_dir / 'label_encoders.pkl'
    if encoders_path.exists():
        try:
            label_encoders = joblib.load(encoders_path)
            print(f"✅ Loaded label encoders: {list(label_encoders.keys())}")
        except Exception as e:
            print(f"⚠️ Failed to load encoders: {e}")

    # Load model
    for model_file in model_files:
        if model_file.exists():
            try:
                print(f"🔄 Attempting to load {model_file.name}...")

                # Try joblib first
                try:
                    loaded_model = joblib.load(model_file)
                    print(f"📦 Loaded with joblib. Type: {type(loaded_model)}")
                except Exception as joblib_error:
                    print(f"⚠️ Joblib failed: {joblib_error}")
                    # Try pickle as fallback
                    try:
                        with open(model_file, 'rb') as f:
                            loaded_model = pickle.load(f)
                        print(f"📦 Loaded with pickle. Type: {type(loaded_model)}")
                    except Exception as pickle_error:
                        print(f"❌ Pickle also failed: {pickle_error}")
                        continue

                # Skip if it's just predictions array (like BESTLOANPRED.pkl)
                if isinstance(loaded_model, np.ndarray):
                    print(f"❌ {model_file.name} contains predictions array, not a model")
                    continue

                # Validate the loaded model
                if validate_model(loaded_model):
                    model = loaded_model
                    print(f"✅ Successfully loaded and validated model from {model_file.name}")

                    # Try to get feature information
                    try:
                        if hasattr(model, 'feature_names_in_'):
                            model_features = list(model.feature_names_in_)
                            print(f"📋 Model expects {len(model_features)} features: {model_features}")
                        elif hasattr(model, 'n_features_in_'):
                            print(f"📋 Model expects {model.n_features_in_} features")
                            # Create default feature names
                            model_features = [
                                                 'person_age', 'person_gender', 'person_education', 'person_income',
                                                 'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
                                                 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                                                 'credit_score', 'previous_loan_defaults_on_file', 'debt_to_income',
                                                 'income_per_age'
                                             ][:model.n_features_in_]
                        else:
                            print("📋 Could not determine expected feature count")
                    except Exception as e:
                        print(f"⚠️ Error getting feature info: {e}")

                    return True
                else:
                    print(f"❌ Model validation failed for {model_file.name}")
                    continue

            except Exception as e:
                print(f"❌ Failed to load {model_file.name}: {e}")
                continue

    print("❌ No valid model file found")
    return False


def recreate_model():
    """Recreate model if the original files are missing or corrupted"""
    global model, label_encoders, model_features

    script_dir = Path(__file__).parent.absolute()
    data_path = script_dir / 'loan_data.csv'

    if not data_path.exists():
        print("❌ loan_data.csv not found - cannot recreate model")
        return False

    try:
        print("\n🛠️ Recreating loan prediction model...")

        # Load and preprocess data
        df = pd.read_csv(data_path)
        print(f"📊 Loaded data with shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")

        # Feature engineering - BEFORE encoding
        df['debt_to_income'] = df['loan_amnt'] / df['person_income']
        df['income_per_age'] = df['person_income'] / (df['person_age'] + 1)

        # Label encoding - store encoders for later use
        categorical_columns = ['person_gender', 'person_education', 'person_home_ownership',
                               'loan_intent', 'previous_loan_defaults_on_file']

        label_encoders = {}  # Reset encoders
        for col in categorical_columns:
            if col in df.columns:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col])
                print(
                    f"🔄 Encoded {col}: {dict(zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)))}")

        # Apply feature scaling (matching your original code)
        df['debt_to_income'] *= 2
        df['person_income'] *= 1.5
        df['previous_loan_defaults_on_file'] *= 2
        df['loan_amnt'] *= 1.2
        df['loan_intent'] *= 1.2
        df['income_per_age'] *= 1.5

        # Prepare data
        X = df.drop("loan_status", axis=1)
        y = df["loan_status"]

        print(f"📊 Features shape: {X.shape}")
        print(f"📊 Feature names: {list(X.columns)}")
        model_features = list(X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

        print("🏃 Training model...")
        model.fit(X_train, y_train)

        # Test the model
        test_score = model.score(X_test, y_test)
        print(f"📈 Model accuracy on test set: {test_score:.3f}")

        # Validate with actual test data
        sample_features = X_test.iloc[:1].values
        if validate_model(model, sample_features):
            # Save model
            model_path = script_dir / 'loan_model_recreated.pkl'
            joblib.dump(model, model_path)
            print(f"💾 Model saved to {model_path}")

            # Save encoders separately
            encoders_path = script_dir / 'label_encoders.pkl'
            joblib.dump(label_encoders, encoders_path)
            print(f"💾 Encoders saved to {encoders_path}")

            print("✅ Successfully recreated and validated loan prediction model!")
            return True
        else:
            print("❌ Model recreation succeeded but validation failed")
            return False

    except Exception as e:
        print(f"❌ Failed to recreate model: {e}")
        traceback.print_exc()
        return False


# Load model at startup
print("=" * 50)
print("💰 LOAN APPROVAL PREDICTION API INITIALIZATION")
print("=" * 50)

model_loaded = load_model_safe()

if not model_loaded:
    print("\n⚠️ Model loading failed - attempting to recreate...")
    if recreate_model():
        model_loaded = load_model_safe()


@app.route('/predict', methods=['POST'])
def predict_loan_approval():
    """Predict loan approval based on input features"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'solution': 'Try the /reload endpoint or check server logs'
        }), 500

    try:
        # Get and validate data
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        # Required fields based on the original dataset
        required_fields = [
            'person_age', 'person_gender', 'person_education', 'person_income',
            'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score', 'previous_loan_defaults_on_file'
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing fields: {missing_fields}',
                'required_fields': required_fields
            }), 400

        # Prepare features
        features = prepare_loan_features(data)
        features_array = np.array(features).reshape(1, -1)

        print(f"🔍 Prepared features shape: {features_array.shape}")
        print(f"🔍 Expected features: {len(model_features) if model_features else 'unknown'}")

        # Predict
        prediction = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0]

        print(f"🎯 Prediction: {prediction}")
        print(f"🎯 Probabilities: {prediction_proba}")

        # Get confidence scores
        approval_probability = float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.5
        rejection_probability = float(prediction_proba[0]) if len(prediction_proba) > 1 else 0.5

        return jsonify({
            'success': True,
            'loan_approved': bool(prediction),
            'approval_probability': approval_probability,
            'rejection_probability': rejection_probability,
            'confidence': max(approval_probability, rejection_probability),
            'risk_assessment': get_risk_assessment(data, approval_probability),
            'features_used': len(features),
            'model_type': str(type(model).__name__)
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 400


def prepare_loan_features(data):
    """Prepare features from input data matching training preprocessing"""
    try:
        # Create a DataFrame for consistent processing
        df_row = pd.DataFrame([data])

        # Feature engineering - same as training
        df_row['debt_to_income'] = df_row['loan_amnt'] / df_row['person_income']
        df_row['income_per_age'] = df_row['person_income'] / (df_row['person_age'] + 1)

        # Apply label encoding using stored encoders
        categorical_columns = ['person_gender', 'person_education', 'person_home_ownership',
                               'loan_intent', 'previous_loan_defaults_on_file']

        for col in categorical_columns:
            if col in label_encoders:
                try:
                    df_row[col] = label_encoders[col].transform(df_row[col])
                except ValueError as e:
                    print(f"⚠️ Unknown category for {col}: {df_row[col].iloc[0]}")
                    # Use the first class as default for unknown categories
                    df_row[col] = 0
            else:
                print(f"⚠️ No encoder found for {col}, using default mapping")
                # Fallback mappings
                if col == 'person_gender':
                    gender_map = {'Male': 1, 'Female': 0, 'male': 1, 'female': 0}
                    df_row[col] = gender_map.get(str(df_row[col].iloc[0]).title(), 0)
                elif col == 'person_education':
                    edu_map = {'High School': 3, 'Associate': 0, 'Bachelor': 1, 'Master': 4, 'Doctorate': 2}
                    df_row[col] = edu_map.get(df_row[col].iloc[0], 0)
                elif col == 'person_home_ownership':
                    home_map = {'RENT': 3, 'MORTGAGE': 0, 'OWN': 2, 'OTHER': 1}
                    df_row[col] = home_map.get(df_row[col].iloc[0], 0)
                elif col == 'loan_intent':
                    intent_map = {'DEBTCONSOLIDATION': 0, 'EDUCATION': 1, 'HOMEIMPROVEMENT': 2,
                                  'MEDICAL': 3, 'PERSONAL': 4, 'VENTURE': 5}
                    df_row[col] = intent_map.get(df_row[col].iloc[0], 0)
                elif col == 'previous_loan_defaults_on_file':
                    df_row[col] = 1 if str(df_row[col].iloc[0]).lower() in ['yes', 'y', '1', 'true'] else 0

        # Apply feature scaling (matching training)
        df_row['debt_to_income'] *= 2
        df_row['person_income'] *= 1.5
        df_row['previous_loan_defaults_on_file'] *= 2
        df_row['loan_amnt'] *= 1.2
        df_row['loan_intent'] *= 1.2
        df_row['income_per_age'] *= 1.5

        # Return features in the expected order
        feature_order = [
            'person_age', 'person_gender', 'person_education', 'person_income',
            'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score', 'previous_loan_defaults_on_file', 'debt_to_income',
            'income_per_age'
        ]

        features = []
        for feature in feature_order:
            if feature in df_row.columns:
                features.append(float(df_row[feature].iloc[0]))
            else:
                print(f"⚠️ Missing feature: {feature}")
                features.append(0.0)

        return features

    except Exception as e:
        print(f"Error in feature preparation: {e}")
        traceback.print_exc()
        raise e


def get_risk_assessment(data, approval_prob):
    """Generate risk assessment based on input data and prediction probability"""
    risk_factors = []

    # Check various risk factors
    debt_to_income_ratio = float(data['loan_amnt']) / float(data['person_income'])
    if debt_to_income_ratio > 0.5:
        risk_factors.append("High debt-to-income ratio")

    if float(data['loan_percent_income']) > 0.3:
        risk_factors.append("Loan represents large portion of income")

    if float(data['loan_int_rate']) > 15:
        risk_factors.append("High interest rate")

    if str(data.get('previous_loan_defaults_on_file', '')).lower() in ['yes', 'y', '1', 'true']:
        risk_factors.append("Previous loan defaults")

    if float(data['cb_person_cred_hist_length']) < 3:
        risk_factors.append("Short credit history")

    if float(data.get('credit_score', 700)) < 600:
        risk_factors.append("Low credit score")

    # Determine overall risk level
    if approval_prob > 0.8:
        risk_level = "Low Risk"
    elif approval_prob > 0.6:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return {
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'debt_to_income_ratio': round(debt_to_income_ratio, 3)
    }


@app.route('/health', methods=['GET'])
def health_check():
    """System health check"""
    model_info = {
        'model_loaded': model is not None,
        'model_type': str(type(model)) if model else None,
        'has_predict': hasattr(model, 'predict') if model else False,
        'has_predict_proba': hasattr(model, 'predict_proba') if model else False,
        'feature_count': len(model_features) if model_features else None,
        'encoders_loaded': bool(label_encoders),
        'encoder_keys': list(label_encoders.keys()) if label_encoders else None
    }

    return jsonify({
        'status': 'running',
        'service': 'Loan Approval Prediction',
        'model_info': model_info
    })


@app.route('/reload', methods=['POST'])
def reload_model():
    """Reload model"""
    global model
    model_loaded = load_model_safe()

    return jsonify({
        'success': model_loaded,
        'model_loaded': model_loaded,
        'model_type': str(type(model)) if model else None,
        'message': 'Model reloaded' if model_loaded else 'Failed to reload model'
    })


@app.route('/recreate-model', methods=['POST'])
def recreate_model_endpoint():
    """Endpoint to recreate model files"""
    success = recreate_model()

    if success:
        # Reload the newly created model
        model_loaded = load_model_safe()
        return jsonify({
            'success': True,
            'message': 'Successfully recreated and loaded model',
            'model_loaded': model_loaded,
            'model_type': str(type(model)) if model else None
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to recreate model',
            'solution': 'Ensure loan_data.csv exists in the same directory'
        }), 500


@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check model state"""
    debug_data = {
        'model_loaded': model is not None,
        'model_type': str(type(model)) if model else None,
        'model_features': model_features,
        'feature_count': len(model_features) if model_features else None,
        'label_encoders': list(label_encoders.keys()) if label_encoders else None,
        'encoder_classes': {k: list(v.classes_) for k, v in label_encoders.items()} if label_encoders else None
    }

    return jsonify(debug_data)


@app.route('/sample-request', methods=['GET'])
def sample_request():
    """Return a sample request format for testing"""
    sample = {
        "person_age": 25,
        "person_gender": "Male",
        "person_education": "Bachelor",
        "person_income": 50000,
        "person_emp_exp": 3,
        "person_home_ownership": "RENT",
        "loan_amnt": 10000,
        "loan_intent": "PERSONAL",
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.2,
        "cb_person_cred_hist_length": 5,
        "credit_score": 650,
        "previous_loan_defaults_on_file": "No"
    }

    return jsonify({
        'sample_request': sample,
        'usage': 'POST this JSON to /predict endpoint'
    })


if __name__ == '__main__':
    print("\nAvailable endpoints:")
    print("  POST /predict - Make loan approval predictions")
    print("  GET  /health - Check system status")
    print("  POST /reload - Reload model")
    print("  POST /recreate-model - Recreate model files")
    print("  GET  /debug - Debug model information")
    print("  GET  /sample-request - Get sample request format")
    print("\nStarting server on http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)