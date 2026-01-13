from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np
import joblib
import json
import pandas as pd
import mindspore as ms
from mindspore import Tensor, context
from datetime import datetime
import os
import model as model_module  # Import the MindSpore model definition

# Initialize FastAPI
app = FastAPI(
    title="Tech-Trust AI API",
    description="API for evaluating developer trust scores based on GitHub activity and credentials",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================
# CREDENTIAL VERIFIER
# ===========================

class CredentialVerifier:
    """Handles credential verification and scoring"""
    
    def __init__(self, credentials_csv='credentials.csv'):
        if not os.path.exists(credentials_csv):
            print(f"⚠ Warning: {credentials_csv} not found - credential verification disabled")
            self.credentials_df = None
            self.enabled = False
        else:
            self.credentials_df = pd.read_csv(credentials_csv)
            self.enabled = True
            print(f"✓ Loaded {len(self.credentials_df)} credentials from database")
        
        self.weight_multipliers = {
            'Easy': 1.0,
            'Medium': 1.2,
            'Hard': 1.5
        }
        
    def verify_credential(self, credential_name, vendor=None):
        """Verify if a credential exists in the database"""
        if not self.enabled:
            return None
            
        query = self.credentials_df['name'] == credential_name
        if vendor:
            query &= self.credentials_df['vendor'] == vendor
        
        result = self.credentials_df[query]
        if len(result) > 0:
            return result.iloc[0].to_dict()
        return None
    
    def calculate_credential_score(self, credentials_list):
        """Calculate weighted credential score"""
        if not self.enabled or not credentials_list or len(credentials_list) == 0:
            return 0.0
        
        total_score = 0
        verified_count = 0
        
        for cred in credentials_list:
            cred_info = self.verify_credential(cred.get('name'), cred.get('vendor'))
            if cred_info:
                base_weight = cred_info['weight']
                difficulty = cred_info['verification_difficulty']
                multiplier = self.weight_multipliers.get(difficulty, 1.0)
                
                total_score += base_weight * multiplier
                verified_count += 1
        
        if verified_count == 0:
            return 0.0
        
        avg_score = total_score / verified_count
        normalized = min(avg_score / 15.0 * 10, 10)
        
        return normalized
    
    def get_credential_breakdown(self, credentials_list):
        """Get detailed breakdown of credentials by category"""
        if not self.enabled:
            return {}
            
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


# Set MindSpore context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # Change to "GPU" if GPU available

# Load model, scaler, and credential verifier on startup
try:
    # Load MindSpore model checkpoint
    model_path = 'trust_model_with_credentials.ckpt'
    scaler_path = 'scaler.joblib'
    
    # Determine input dimension (9 features)
    input_dim = 9
    trust_model = model_module.TrustScoreModel(input_dim)
    
    # Load checkpoint
    param_dict = ms.load_checkpoint(model_path)
    ms.load_param_into_net(trust_model, param_dict)
    trust_model.set_train(False)  # Set to evaluation mode
    
    scaler = joblib.load(scaler_path)
    credential_verifier = CredentialVerifier('credentials.csv')
    
    # Load metrics if available
    try:
        with open('model_metrics.json', 'r') as f:
            model_info = json.load(f)
    except:
        model_info = {}
    
    print("✓ MindSpore model loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load model - {e}")
    trust_model = None
    scaler = None
    credential_verifier = CredentialVerifier('credentials.csv')
    model_info = {}


# ===========================
# REQUEST/RESPONSE MODELS
# ===========================

class Credential(BaseModel):
    name: str = Field(..., description="Credential name (must match credentials database)")
    vendor: Optional[str] = Field(None, description="Credential vendor/issuer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "AWS Certified Solutions Architect",
                "vendor": "AWS"
            }
        }


class DeveloperProfile(BaseModel):
    username: str = Field(..., description="GitHub username")
    total_stars: int = Field(..., ge=0, description="Total stars across all repos")
    total_forks: int = Field(..., ge=0, description="Total forks across all repos")
    total_issues: int = Field(..., ge=0, description="Total issues created")
    total_prs: int = Field(..., ge=0, description="Total pull requests")
    total_contributors: int = Field(..., ge=0, description="Total contributors across repos")
    languages: List[str] = Field(..., description="List of programming languages used")
    repo_count: int = Field(..., ge=0, description="Number of repositories")
    credentials: Optional[List[Credential]] = Field(default=[], description="List of verified credentials")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "octocat",
                "total_stars": 1500,
                "total_forks": 300,
                "total_issues": 50,
                "total_prs": 120,
                "total_contributors": 25,
                "languages": ["Python", "JavaScript", "Go"],
                "repo_count": 15,
                "credentials": [
                    {"name": "AWS Certified Solutions Architect", "vendor": "AWS"},
                    {"name": "CISSP", "vendor": "ISC2"}
                ]
            }
        }


class TrustScoreResponse(BaseModel):
    username: str
    trust_score: float = Field(..., description="Final trust score from 0-10")
    github_score: float = Field(..., description="GitHub activity score from 0-10")
    credential_score: float = Field(..., description="Credential verification score from 0-10")
    confidence_level: str = Field(..., description="Low, Medium, or High")
    breakdown: dict = Field(..., description="Score breakdown by category")
    credentials_info: dict = Field(..., description="Credential verification details")
    timestamp: str


class BatchRequest(BaseModel):
    developers: List[DeveloperProfile]


class BatchResponse(BaseModel):
    results: List[TrustScoreResponse]
    total_processed: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    credentials_enabled: bool
    model_info: dict


# ===========================
# HELPER FUNCTIONS
# ===========================

def calculate_features(profile: DeveloperProfile):
    """Calculate features from developer profile"""
    
    # Language weights (same as training)
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
    
    # Calculate base scores
    popularity_score = profile.total_stars + profile.total_forks
    activity_score = profile.total_prs + profile.total_issues
    collab_score = profile.total_contributors
    tech_diversity = len(set(profile.languages)) if profile.languages else 0
    
    language_score = np.mean([
        language_weights.get(lang, 1) for lang in profile.languages
    ]) if profile.languages else 0
    
    # Calculate GitHub score (0-10)
    # Normalize using simple scaling
    github_score = (
        0.35 * min(popularity_score / 10000, 1) +
        0.30 * min(activity_score / 5000, 1) +
        0.20 * min(collab_score / 500, 1) +
        0.10 * min(tech_diversity / 10, 1) +
        0.05 * min(language_score / 5, 1)
    ) * 10
    
    # Process credentials
    credentials_list = [cred.dict() for cred in profile.credentials] if profile.credentials else []
    credential_score = credential_verifier.calculate_credential_score(credentials_list)
    credential_count = len(credentials_list)
    
    # Calculate credential diversity
    verified_categories = set()
    for cred in credentials_list:
        cred_info = credential_verifier.verify_credential(cred.get('name'), cred.get('vendor'))
        if cred_info:
            verified_categories.add(cred_info.get('category', 'Unknown'))
    credential_diversity = len(verified_categories)
    
    return {
        'popularity_score': popularity_score,
        'activity_score': activity_score,
        'collab_score': collab_score,
        'tech_diversity': tech_diversity,
        'language_score': language_score,
        'repo_count': profile.repo_count,
        'credential_score': credential_score,
        'credential_count': credential_count,
        'credential_diversity': credential_diversity,
        'github_score': github_score,
        'credentials_list': credentials_list
    }


def get_confidence_level(score: float) -> str:
    """Determine confidence level based on score"""
    if score >= 7.0:
        return "High"
    elif score >= 4.0:
        return "Medium"
    else:
        return "Low"


def predict_trust_score(profile: DeveloperProfile):
    """Predict trust score for a developer"""
    
    if trust_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate features
    features_dict = calculate_features(profile)
    
    # Prepare features array in correct order (must match training)
    # Note: github_score is NOT used in training, only the 9 features below
    features = np.array([[
        features_dict['popularity_score'],
        features_dict['activity_score'],
        features_dict['collab_score'],
        features_dict['tech_diversity'],
        features_dict['language_score'],
        features_dict['repo_count'],
        features_dict['credential_score'],
        features_dict['credential_count'],
        features_dict['credential_diversity']
    ]], dtype=np.float32)
    
    # Scale features
    features_scaled = scaler.transform(features).astype(np.float32)
    
    # Predict using MindSpore model
    features_tensor = Tensor(features_scaled)
    trust_model.set_train(False)  # Ensure evaluation mode
    prediction = trust_model(features_tensor)
    trust_score = float(np.clip(prediction.asnumpy()[0][0], 0, 10))
    
    # Calculate breakdown percentages for display
    github_weight = 60
    credential_weight = 35
    diversity_weight = 5
    
    breakdown = {
        "github_activity": {
            "score": round(features_dict['github_score'], 2),
            "weight": f"{github_weight}%",
            "components": {
                "popularity": round(features_dict['popularity_score'], 0),
                "activity": round(features_dict['activity_score'], 0),
                "collaboration": round(features_dict['collab_score'], 0),
                "tech_diversity": round(features_dict['tech_diversity'], 0),
                "language_expertise": round(features_dict['language_score'], 2)
            }
        },
        "credentials": {
            "score": round(features_dict['credential_score'], 2),
            "weight": f"{credential_weight}%",
            "count": features_dict['credential_count']
        },
        "diversity_bonus": {
            "weight": f"{diversity_weight}%",
            "credential_categories": features_dict['credential_diversity']
        }
    }
    
    # Get verified credentials info
    verified_credentials = []
    unverified_credentials = []
    credential_breakdown = credential_verifier.get_credential_breakdown(features_dict['credentials_list'])
    
    for cred in features_dict['credentials_list']:
        cred_info = credential_verifier.verify_credential(cred.get('name'), cred.get('vendor'))
        if cred_info:
            verified_credentials.append({
                'name': cred.get('name'),
                'vendor': cred.get('vendor'),
                'category': cred_info.get('category'),
                'weight': cred_info.get('weight'),
                'difficulty': cred_info.get('verification_difficulty')
            })
        else:
            unverified_credentials.append(cred.get('name'))
    
    credentials_info = {
        "total_submitted": features_dict['credential_count'],
        "verified_count": len(verified_credentials),
        "unverified_count": len(unverified_credentials),
        "verified_credentials": verified_credentials,
        "unverified_credentials": unverified_credentials,
        "category_breakdown": credential_breakdown
    }
    
    return TrustScoreResponse(
        username=profile.username,
        trust_score=round(trust_score, 2),
        github_score=round(features_dict['github_score'], 2),
        credential_score=round(features_dict['credential_score'], 2),
        confidence_level=get_confidence_level(trust_score),
        breakdown=breakdown,
        credentials_info=credentials_info,
        timestamp=datetime.now().isoformat()
    )


# ===========================
# API ENDPOINTS
# ===========================

@app.get("/", response_model=dict)
async def root():
    """API root endpoint"""
    return {
        "message": "Tech-Trust AI API - Comprehensive Developer Trust Scoring",
        "version": "2.0.0",
        "features": [
            "GitHub activity analysis",
            "Professional credential verification",
            "Dual-component trust scoring (60% GitHub + 35% Credentials + 5% Diversity)",
            "Powered by Huawei MindSpore"
        ],
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "batch": "/api/v1/predict/batch",
            "credentials": "/api/v1/credentials",
            "metrics": "/api/v1/metrics",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if trust_model is not None else "model_not_loaded",
        model_loaded=trust_model is not None,
        credentials_enabled=credential_verifier.enabled if credential_verifier else False,
        model_info=model_info
    )


@app.post("/api/v1/predict", response_model=TrustScoreResponse)
async def predict(profile: DeveloperProfile):
    """
    Predict comprehensive trust score for a single developer
    
    The trust score combines:
    - **60%** GitHub activity (stars, forks, PRs, issues, collaboration)
    - **35%** Verified credentials (certifications, licenses, degrees)
    - **5%** Diversity bonus (variety of credential categories)
    
    ## Parameters:
    - **username**: GitHub username
    - **total_stars**: Sum of stars across all repositories
    - **total_forks**: Sum of forks across all repositories
    - **total_issues**: Total number of issues created
    - **total_prs**: Total number of pull requests
    - **total_contributors**: Total contributors across all repos
    - **languages**: List of programming languages used
    - **repo_count**: Number of repositories owned
    - **credentials**: List of professional credentials (optional)
    
    ## Returns:
    - Final trust score (0-10)
    - GitHub activity score
    - Credential verification score
    - Detailed breakdown
    - Verified vs unverified credentials
    """
    try:
        return predict_trust_score(profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """
    Predict trust scores for multiple developers
    
    Send an array of developer profiles and get back trust scores for all
    """
    try:
        results = []
        for profile in request.developers:
            result = predict_trust_score(profile)
            results.append(result)
        
        return BatchResponse(
            results=results,
            total_processed=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/credentials", response_model=dict)
async def list_credentials():
    """
    Get list of all supported credentials in the database
    
    Returns information about certifications, licenses, and degrees that can be verified
    """
    if not credential_verifier.enabled:
        raise HTTPException(status_code=503, detail="Credential verification not available")
    
    credentials_list = credential_verifier.credentials_df.to_dict('records')
    
    # Group by category
    by_category = {}
    for cred in credentials_list:
        category = cred['category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append({
            'name': cred['name'],
            'vendor': cred['vendor'],
            'weight': cred['weight'],
            'type': cred['type'],
            'difficulty': cred['verification_difficulty']
        })
    
    return {
        "total_credentials": len(credentials_list),
        "categories": list(by_category.keys()),
        "credentials_by_category": by_category
    }


@app.get("/api/v1/metrics", response_model=dict)
async def get_metrics():
    """Get model training metrics and information"""
    if not model_info:
        raise HTTPException(status_code=404, detail="Model metrics not found")
    return model_info


# ===========================
# RUN SERVER
# ===========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )