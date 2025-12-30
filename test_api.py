"""
Test script for Tech-Trust AI API
Run this after starting the API server to verify everything works
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health():
    """Test the health endpoint"""
    print_section("Testing Health Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"API Status: {data['status']}")
        print(f"Model Loaded: {data['model_loaded']}")
        
        if data.get('model_info'):
            print(f"\nModel Metrics:")
            print(f"  MAE: {data['model_info'].get('mae', 'N/A')}")
            print(f"  R² Score: {data['model_info'].get('r2_score', 'N/A')}")
            print(f"  Trained On: {data['model_info'].get('trained_on', 'N/A')}")
        
        return response.status_code == 200
    except Exception as e:
        print(f" Error: {e}")
        return False

def test_single_prediction():
    """Test single developer prediction"""
    print_section("Testing Single Prediction")
    
    # Test developer profile
    developer = {
        "username": "test_developer",
        "total_stars": 1500,
        "total_forks": 300,
        "total_issues": 50,
        "total_prs": 120,
        "total_contributors": 25,
        "languages": ["Python", "JavaScript", "Go"],
        "repo_count": 15
    }
    
    print("\nInput Data:")
    print(json.dumps(developer, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/predict",
            json=developer
        )
        data = response.json()
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"\nPrediction Results:")
        print(f"  Username: {data['username']}")
        print(f"  Trust Score: {data['trust_score']}/10")
        print(f"  Confidence Level: {data['confidence_level']}")
        
        print(f"\n  Score Breakdown:")
        for key, value in data['breakdown'].items():
            print(f"    {key.replace('_', ' ').title()}: {value}%")
        
        print(f"\n  Timestamp: {data['timestamp']}")
        
        return response.status_code == 200
    except Exception as e:
        print(f" Error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction"""
    print_section("Testing Batch Prediction")
    
    # Test multiple developers
    developers = {
        "developers": [
            {
                "username": "junior_dev",
                "total_stars": 200,
                "total_forks": 30,
                "total_issues": 10,
                "total_prs": 25,
                "total_contributors": 5,
                "languages": ["Python", "JavaScript"],
                "repo_count": 5
            },
            {
                "username": "senior_dev",
                "total_stars": 5000,
                "total_forks": 1000,
                "total_issues": 200,
                "total_prs": 500,
                "total_contributors": 100,
                "languages": ["Python", "Go", "Rust", "TypeScript", "C++"],
                "repo_count": 30
            },
            {
                "username": "beginner_dev",
                "total_stars": 10,
                "total_forks": 2,
                "total_issues": 3,
                "total_prs": 5,
                "total_contributors": 1,
                "languages": ["JavaScript"],
                "repo_count": 2
            }
        ]
    }
    
    print(f"\nTesting with {len(developers['developers'])} developers...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/batch",
            json=developers
        )
        data = response.json()
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Total Processed: {data['total_processed']}")
        
        print("\nResults:")
        for result in data['results']:
            print(f"\n   {result['username']}")
            print(f"     Score: {result['trust_score']}/10 ({result['confidence_level']})")
        
        return response.status_code == 200
    except Exception as e:
        print(f" Error: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print_section("Testing Metrics Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/metrics")
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"\nModel Performance:")
        print(f"  MAE (Mean Absolute Error): {data.get('mae', 'N/A')}")
        print(f"  RMSE (Root Mean Squared Error): {data.get('rmse', 'N/A')}")
        print(f"  R² Score: {data.get('r2_score', 'N/A')}")
        print(f"\nTraining Info:")
        print(f"  Training Samples: {data.get('training_samples', 'N/A')}")
        print(f"  Test Samples: {data.get('test_samples', 'N/A')}")
        print(f"  Epochs Completed: {data.get('epochs_completed', 'N/A')}")
        print(f"  Trained On: {data.get('trained_on', 'N/A')}")
        
        return response.status_code == 200
    except Exception as e:
        print(f" Error: {e}")
        return False

def test_error_handling():
    """Test API error handling"""
    print_section("Testing Error Handling")
    
    # Test with missing required fields
    invalid_data = {
        "username": "test"
        # Missing all other required fields
    }
    
    print("\nTesting with invalid data (missing fields)...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/predict",
            json=invalid_data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 422:
            print("✓ API correctly rejected invalid input")
            return True
        else:
            print(" Expected 422 error for invalid input")
            return False
    except Exception as e:
        print(f" Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  TECH-TRUST API TEST SUITE")
    print("="*60)
    print(f"Testing API at: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Metrics Endpoint", test_metrics),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    print("\n  Make sure the API server is running first!")
    print("Start it with: python api.py")
    print("Or: uvicorn api:app --reload\n")
    
    input("Press Enter to start tests...")
    
    main()