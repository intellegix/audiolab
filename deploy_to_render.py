#!/usr/bin/env python3
"""
AudioLab Render Deployment Script
Deploy AudioLab to Render.io using the REST API
"""

import requests
import json
import time
import os
from typing import Dict, Optional

class RenderDeployer:
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.render.com/v1"
        self.api_key = api_key or os.environ.get('RENDER_API_KEY')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_user_info(self):
        """Get user information to verify API key"""
        response = requests.get(f"{self.base_url}/users/me", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def create_service_from_repo(self, repo_url: str = "https://github.com/intellegix/audiolab"):
        """Create services from GitHub repository with render.yaml"""

        # Service configuration based on our render.yaml
        services_config = {
            "services": [
                {
                    "type": "web_service",
                    "name": "audiolab-api",
                    "repo": repo_url,
                    "plan": "standard",
                    "buildCommand": "pip install -r requirements.txt",
                    "startCommand": "uvicorn src.main:app --host 0.0.0.0 --port $PORT --workers 4",
                    "healthCheckPath": "/health",
                    "envVars": [
                        {"key": "ENVIRONMENT", "value": "production"},
                        {"key": "CORS_ORIGINS", "value": "*"},
                        {"key": "AUDIO_OUTPUT_PATH", "value": "/tmp/audiolab/output"}
                    ]
                }
            ]
        }

        print("🚀 Creating AudioLab services on Render...")

        # Create web service
        web_service_data = {
            "type": "web_service",
            "name": "audiolab-api",
            "repo": repo_url,
            "branch": "master",
            "buildCommand": "pip install -r requirements.txt",
            "startCommand": "uvicorn src.main:app --host 0.0.0.0 --port $PORT --workers 4",
            "plan": "standard",
            "region": "ohio",
            "healthCheckPath": "/health",
            "envVars": [
                {"key": "ENVIRONMENT", "value": "production"},
                {"key": "CORS_ORIGINS", "value": "*"},
                {"key": "AUDIO_OUTPUT_PATH", "value": "/tmp/audiolab/output"}
            ]
        }

        response = requests.post(
            f"{self.base_url}/services",
            headers=self.headers,
            json=web_service_data
        )

        if response.status_code == 201:
            service = response.json()
            print(f"✅ Web service created: {service.get('name')}")
            print(f"   Service ID: {service.get('id')}")
            print(f"   URL: https://{service.get('name')}.onrender.com")
            return service
        else:
            print(f"❌ Failed to create web service: {response.status_code}")
            print(f"   Error: {response.text}")
            return None

    def create_database(self, database_name: str = "audiolab-db"):
        """Create PostgreSQL database"""
        database_data = {
            "name": database_name,
            "plan": "free",
            "region": "ohio",
            "databaseName": "audiolab",
            "databaseUser": "audiolab_user"
        }

        response = requests.post(
            f"{self.base_url}/postgres",
            headers=self.headers,
            json=database_data
        )

        if response.status_code == 201:
            database = response.json()
            print(f"✅ Database created: {database.get('name')}")
            return database
        else:
            print(f"❌ Failed to create database: {response.status_code}")
            print(f"   Error: {response.text}")
            return None

    def create_redis(self, redis_name: str = "audiolab-redis"):
        """Create Redis cache service"""
        redis_data = {
            "name": redis_name,
            "plan": "free",
            "region": "ohio"
        }

        response = requests.post(
            f"{self.base_url}/redis",
            headers=self.headers,
            json=redis_data
        )

        if response.status_code == 201:
            redis = response.json()
            print(f"✅ Redis created: {redis.get('name')}")
            return redis
        else:
            print(f"❌ Failed to create Redis: {response.status_code}")
            print(f"   Error: {response.text}")
            return None

    def get_service_status(self, service_id: str):
        """Get service deployment status"""
        response = requests.get(f"{self.base_url}/services/{service_id}", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return None

    def monitor_deployment(self, service_id: str, timeout: int = 900):
        """Monitor deployment progress"""
        print("📊 Monitoring deployment progress...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            service = self.get_service_status(service_id)
            if service:
                status = service.get('status', 'unknown')
                print(f"   Status: {status}")

                if status == 'live':
                    print("🎉 Deployment successful!")
                    return True
                elif status == 'failed':
                    print("❌ Deployment failed!")
                    return False

            time.sleep(30)  # Check every 30 seconds

        print("⏰ Deployment monitoring timed out")
        return False

def main():
    print("🔧 AudioLab Render Deployment Tool")
    print("=" * 50)

    # Check for API key
    api_key = os.environ.get('RENDER_API_KEY')
    if not api_key:
        print("❌ RENDER_API_KEY environment variable not found")
        print("   Please set your Render API key:")
        print("   export RENDER_API_KEY=your_api_key_here")
        print("   Get your API key from: https://dashboard.render.com/account/api-keys")
        return False

    # Initialize deployer
    deployer = RenderDeployer(api_key)

    # Verify API key
    user_info = deployer.get_user_info()
    if not user_info:
        print("❌ Invalid API key or connection failed")
        return False

    print(f"✅ Connected as: {user_info.get('email', 'Unknown')}")

    # Create services
    print("\n🚀 Creating Render services...")

    # Create database first
    database = deployer.create_database()
    if not database:
        return False

    # Create Redis
    redis = deployer.create_redis()
    if not redis:
        return False

    # Create web service
    web_service = deployer.create_service_from_repo()
    if not web_service:
        return False

    # Monitor deployment
    service_id = web_service.get('id')
    if service_id:
        success = deployer.monitor_deployment(service_id)
        if success:
            print(f"\n🎉 AudioLab deployed successfully!")
            print(f"   API URL: https://audiolab-api.onrender.com")
            print(f"   Health Check: https://audiolab-api.onrender.com/health")
            print(f"   Documentation: https://audiolab-api.onrender.com/docs")
        return success

    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)