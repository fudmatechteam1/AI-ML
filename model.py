import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ===========================
# CREDENTIAL VERIFICATION SYSTEM
# ===========================

class CredentialVerifier:
    """Handles credential verification and scoring"""
    
    def __init__(self, credentials_csv='credentials.csv'):
        if not os.path.exists(credentials_csv):
            raise FileNotFoundError(
                f"\n❌ Credentials file not found: {credentials_csv}\n"
                f"Please create this file with the following columns:\n"
                f"name,type,category,weight,vendor,verification_url,requires_renewal,verification_difficulty\n"
            )
        self.credentials_df = pd.read_csv(credentials_csv)
        self.weight_multipliers = {
            'Easy': 1.0,
            'Medium': 1.2,
            'Hard': 1.5
        }
        
    def verify_credential(self, credential_name, vendor=None):
        """
        Verify if a credential exists in the database
        Returns: dict with credential info or None
        """
        query = self.credentials_df['name'] == credential_name
        if vendor:
            query &= self.credentials_df['vendor'] == vendor
        
        result = self.credentials_df[query]
        if len(result) > 0:
            return result.iloc[0].to_dict()
        return None
    
    def calculate_credential_score(self, credentials_list):
        """
        Calculate weighted credential score
        credentials_list: list of dicts with 'name' and optional 'vendor'
        """
        if not credentials_list or len(credentials_list) == 0:
            return 0.0
        
        total_score = 0
        verified_count = 0
        
        for cred in credentials_list:
            cred_info = self.verify_credential(cred.get('name'), cred.get('vendor'))
            if cred_info:
                # Base weight from credential
                base_weight = cred_info['weight']
                
                # Multiply by difficulty
                difficulty = cred_info['verification_difficulty']
                multiplier = self.weight_multipliers.get(difficulty, 1.0)
                
                total_score += base_weight * multiplier
                verified_count += 1
        
        # Average score normalized to 0-10 scale
        if verified_count == 0:
            return 0.0
        
        avg_score = total_score / verified_count
        # Normalize: assume max possible is 15 (weight=10 * difficulty=1.5)
        normalized = min(avg_score / 15.0 * 10, 10)
        
        return normalized
    
    def get_credential_breakdown(self, credentials_list):
        """Get detailed breakdown of credentials by category"""
        breakdown = {
            'Cloud': 0,
            'DevOps': 0,
            'Security': 0,
            'Engineering': 0,
            'Architecture': 0,
            'Education': 0,
            'Communications': 0
        }
        
        for cred in credentials_list:
            cred_info = self.verify_credential(cred.get('name'), cred.get('vendor'))
            if cred_info:
                category = cred_info['category']
                if category in breakdown:
                    breakdown[category] += 1
        
        return breakdown


# ===========================
# STEP 1 – LOAD & AGGREGATE DATA (Enhanced)
# ===========================

def load_and_prepare_data(github_file, credentials_file=None):
    print(f"Loading dataset: {github_file}")
    df = pd.read_csv(github_file)

    # Clean language column
    df['language'] = df['language'].fillna("Unknown")

    # Extract GitHub username from "repositories" like: "octocat/hello-world"
    df['username'] = df['repositories'].apply(lambda x: x.split('/')[0])

    # Aggregate per developer
    grouped = df.groupby('username').agg({
        'stars_count': 'sum',
        'forks_count': 'sum',
        'issues_count': 'sum',
        'pull_requests': 'sum',
        'contributors': 'sum',
        'language': lambda x: list(set(x)),
        'repositories': 'count'
    }).reset_index()

    # Rename for clarity
    grouped.columns = [
        'username',
        'total_stars',
        'total_forks',
        'total_issues',
        'total_prs',
        'total_contributors',
        'languages',
        'repo_count'
    ]

    # If credentials file provided, load and merge
    if credentials_file and os.path.exists(credentials_file):
        print(f"Loading developer credentials: {credentials_file}")
        creds_df = pd.read_csv(credentials_file)
        
        # Check if this is a developer-to-credentials mapping file
        if 'username' in creds_df.columns and 'credentials' in creds_df.columns:
            # This is a mapping file: username -> credentials
            grouped = grouped.merge(creds_df, on='username', how='left')
            grouped['credentials'] = grouped['credentials'].fillna('[]')
            print(f"[OK] Mapped credentials for {creds_df.shape[0]} developers")
        else:
            # This is just a credentials database, not a mapping
            print(f"[INFO] File contains {creds_df.shape[0]} credentials but no username mapping")
            print("[INFO] Using empty credentials for all developers")
            grouped['credentials'] = '[]'
    else:
        # Create empty credentials column
        grouped['credentials'] = '[]'

    print(f"[OK] Aggregated into {len(grouped)} developer profiles")

    return grouped



# STEP 2 – FEATURE ENGINEERING 


def engineer_features(df, credential_verifier=None):

    print("Engineering features...")

    # Original features
    df['popularity_score'] = df['total_stars'] + df['total_forks']
    df['activity_score'] = df['total_prs'] + df['total_issues']
    df['collab_score'] = df['total_contributors']
    df['tech_diversity'] = df['languages'].apply(lambda x: len(set(x)))

    # Enhanced language weights
    language_weights = {
        "Python": 5, "JavaScript": 5, "TypeScript": 5,
        "Go": 5, "Rust": 5,
        "Java": 4, "C++": 4, "C#": 4, "C": 4,
        "Kotlin": 4, "Swift": 4,
        "Ruby": 3, "PHP": 3, "Dart": 3,
        "Scala": 3, "Haskell": 3, "Elixir": 3,
        "HTML": 2, "CSS": 2, "SCSS": 2, "SASS": 2,
        "Lua": 2, "R": 3, "Julia": 3,
        "Perl": 2, "Shell": 2, "Objective-C": 2,
        "AutoHotkey": 1, "Unknown": 0
    }

    df['language_score'] = df['languages'].apply(
        lambda langs: np.mean([language_weights.get(lang, 1) for lang in langs])
    )

    # NEW: Credential verification features
    if credential_verifier:
        print("Processing credentials...")
        
        # Parse credentials JSON and calculate scores
        df['credentials_parsed'] = df['credentials'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )
        
        df['credential_score'] = df['credentials_parsed'].apply(
            lambda creds: credential_verifier.calculate_credential_score(creds)
        )
        
        df['credential_count'] = df['credentials_parsed'].apply(len)
        
        # Category diversity (how many different credential categories)
        df['credential_diversity'] = df['credentials_parsed'].apply(
            lambda creds: len(set([
                credential_verifier.verify_credential(c.get('name')).get('category', 'Unknown')
                for c in creds 
                if credential_verifier.verify_credential(c.get('name'))
            ]))
        )
    else:
        df['credential_score'] = 0
        df['credential_count'] = 0
        df['credential_diversity'] = 0

    # Compute Enhanced Trust Score with credentials
    df['trust_score'] = (
        0.25 * normalize(df['popularity_score']) +
        0.20 * normalize(df['activity_score']) +
        0.15 * normalize(df['collab_score']) +
        0.10 * normalize(df['tech_diversity']) +
        0.10 * normalize(df['language_score']) +
        0.15 * normalize(df['credential_score']) +
        0.05 * normalize(df['credential_diversity'])
    ) * 10

    df['trust_score'] = df['trust_score'].clip(0, 10)

    print("[OK] Feature engineering complete")

    return df


def normalize(series):
    """Normalize values between 0-1."""
    return (series - series.min()) / (series.max() - series.min() + 1e-8)


# STEP 3 – BUILD MODEL (Enhanced Architecture)


def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model



# STEP 4 – TRAIN MODEL (Enhanced)


def train(df):

    features = [
        'popularity_score',
        'activity_score',
        'collab_score',
        'tech_diversity',
        'language_score',
        'repo_count',
        'credential_score',
        'credential_count',
        'credential_diversity'
    ]

    X = df[features].values
    y = df['trust_score'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'scaler.joblib')

    # Build model
    model = build_model(X_train.shape[1])

    # Setup callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train with callbacks
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    # Make predictions for additional metrics
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION METRICS")
    print(f"{'='*50}")
    print(f" Test MAE:  {mae:.4f} (Mean Absolute Error)")
    print(f" Test RMSE: {rmse:.4f} (Root Mean Squared Error)")
    print(f" R² Score:  {r2:.4f} (Coefficient of Determination)")
    print(f"{'='*50}\n")

    # Save model in modern format
    model.save("trust_model_with_credentials.keras")
    print("[OK] Model saved as trust_model_with_credentials.keras")

    # Save metrics to JSON
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'test_loss': float(loss),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': features,
        'trained_on': datetime.now().isoformat(),
        'epochs_completed': len(history.history['loss'])
    }
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("[OK] Metrics saved to model_metrics.json")

    return model, metrics


# ===========================
# STEP 5 – PREDICTION INTERFACE
# ===========================

def predict_trust_score(developer_data, model_path='trust_model_with_credentials.keras', 
                       scaler_path='scaler.joblib', credentials_csv='credentials.csv'):
    """
    Predict trust score for a new developer
    
    developer_data: dict with keys matching training features
    Example:
    {
        'popularity_score': 1000,
        'activity_score': 500,
        'collab_score': 50,
        'tech_diversity': 5,
        'language_score': 4.5,
        'repo_count': 20,
        'credentials': [
            {'name': 'AWS Certified Solutions Architect', 'vendor': 'AWS'},
            {'name': 'CISSP', 'vendor': 'ISC2'}
        ]
    }
    """
    # Load model and scaler
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Process credentials
    verifier = CredentialVerifier(credentials_csv)
    credentials = developer_data.get('credentials', [])
    
    # Calculate credential features
    credential_score = verifier.calculate_credential_score(credentials)
    credential_count = len(credentials)
    credential_diversity = len(set([
        verifier.verify_credential(c.get('name')).get('category', 'Unknown')
        for c in credentials 
        if verifier.verify_credential(c.get('name'))
    ]))
    
    # Prepare feature vector
    features = [
        developer_data.get('popularity_score', 0),
        developer_data.get('activity_score', 0),
        developer_data.get('collab_score', 0),
        developer_data.get('tech_diversity', 0),
        developer_data.get('language_score', 0),
        developer_data.get('repo_count', 0),
        credential_score,
        credential_count,
        credential_diversity
    ]
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled, verbose=0)[0][0]
    
    # Get credential breakdown
    breakdown = verifier.get_credential_breakdown(credentials)
    
    return {
        'trust_score': float(np.clip(prediction, 0, 10)),
        'credential_score': float(credential_score),
        'credential_count': credential_count,
        'credential_breakdown': breakdown,
        'verified_credentials': [
            c.get('name') for c in credentials 
            if verifier.verify_credential(c.get('name'))
        ]
    }


# MAIN


def main():
    print("\n" + "="*60)
    print("TECH-TRUST AI MODEL TRAINING (WITH CREDENTIALS)")
    print("="*60 + "\n")

    # Check for required files
    if not os.path.exists("github_dataset.csv"):
        print(" ERROR: github_dataset.csv not found!")
        print("Please ensure your GitHub dataset is in the current directory.")
        return
    
    # Initialize credential verifier (optional)
    credential_verifier = None
    if os.path.exists('credentials.csv'):
        print("Credentials database found - enabling credential verification")
        credential_verifier = CredentialVerifier('credentials.csv')
    else:
        print(" Warning: credentials.csv not found")
        print("  Model will train WITHOUT credential verification features")
        print("  To enable credentials, create credentials.csv with your data\n")
    
    # Check for developer credentials mapping file
    # This should have columns: username, credentials (JSON format)
    credentials_file = None
    for fname in ['developer_credentials.csv', 'comprehensive_credentials_database.csv']:
        if os.path.exists(fname):
            # Peek at the file to see if it has the right format
            temp_df = pd.read_csv(fname, nrows=1)
            if 'username' in temp_df.columns and 'credentials' in temp_df.columns:
                credentials_file = fname
                print(f"✓ Developer credentials mapping found: {fname}")
                break
            else:
                print(f"⚠ Note: {fname} found but doesn't contain 'username' and 'credentials' columns")
                print(f"  Expected format: username,credentials")
                print(f"  File contains columns: {', '.join(temp_df.columns)}\n")
    
    if not credentials_file:
        print(" No developer credentials mapping file found")
        print("  Training with empty credentials for all developers")
        print("  To add credentials, create a CSV with format:")
        print('  username,credentials')
        print('  john_doe,"[{\\"name\\": \\"AWS Certified\\", \\"vendor\\": \\"AWS\\"}]"\n')
    
    df = load_and_prepare_data("github_dataset.csv", credentials_file=credentials_file)
    
    df = engineer_features(df, credential_verifier)
    df.to_csv("developer_training_data_with_credentials.csv", index=False)

    print("[OK] Training dataset saved\n")

    model, metrics = train(df)

    print("="*60)
    print("TRAINING COMPLETE – MODEL READY FOR DEPLOYMENT")
    print("="*60)
    print(f"Model file: trust_model_with_credentials.keras")
    print(f"Scaler file: scaler.joblib")
    print(f"Metrics file: model_metrics.json")
    if credential_verifier:
        print(f"Credentials DB: credentials.csv")
    print("="*60 + "\n")
    
    # Example prediction (only if credentials are available)
    if credential_verifier:
        print("="*60)
        print("EXAMPLE PREDICTION")
        print("="*60)
        
        example_developer = {
            'popularity_score': 1500,
            'activity_score': 800,
            'collab_score': 75,
            'tech_diversity': 6,
            'language_score': 4.8,
            'repo_count': 25,
            'credentials': [
                {'name': 'AWS Certified Solutions Architect', 'vendor': 'AWS'},
                {'name': 'CISSP', 'vendor': 'ISC2'},
                {'name': 'Bachelor of Science in Computer Science'}
            ]
        }
        
        result = predict_trust_score(example_developer)
        print(f"Trust Score: {result['trust_score']:.2f}/10")
        print(f"Credential Score: {result['credential_score']:.2f}/10")
        print(f"Verified Credentials: {', '.join(result['verified_credentials'])}")
        print(f"Credential Breakdown: {result['credential_breakdown']}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()