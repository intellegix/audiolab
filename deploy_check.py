#!/usr/bin/env python3
"""
AudioLab Render Deployment Status Check
"""

import requests
import json
import sys
import os

def main():
    print("AudioLab Render Deployment Tool")
    print("=" * 50)

    # Check if render.yaml exists
    if not os.path.exists("render.yaml"):
        print("[ERROR] render.yaml not found")
        return False

    print("[OK] render.yaml found")

    # Test Render API connectivity
    try:
        response = requests.get("https://api.render.com/v1/services", timeout=10)
        print(f"[OK] Render API accessible (Status: {response.status_code})")
    except Exception as e:
        print(f"[WARNING] Render API test failed: {e}")

    # Show configuration
    print("\nDeployment Configuration:")
    with open("render.yaml", "r") as f:
        content = f.read()
        print(content[:300] + "..." if len(content) > 300 else content)

    print("\nDeployment Steps:")
    print("================")
    print("1. Visit: https://dashboard.render.com/")
    print("2. Click 'New' -> 'Web Service'")
    print("3. Connect GitHub and select 'intellegix/audiolab'")
    print("4. Render auto-detects render.yaml configuration")
    print("5. Click 'Create Web Service'")
    print("")
    print("Expected Services:")
    print("- audiolab-api (Web service)")
    print("- audiolab-db (PostgreSQL)")
    print("- audiolab-redis (Redis)")
    print("")
    print("Expected URL: https://audiolab-api.onrender.com")
    print("Health Check: https://audiolab-api.onrender.com/health")
    print("API Docs: https://audiolab-api.onrender.com/docs")
    print("")
    print("Timeline:")
    print("- Service creation: 1-2 minutes")
    print("- Build process: 8-12 minutes")
    print("- AI model loading: 1-2 minutes")
    print("- Total: ~10-15 minutes")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)