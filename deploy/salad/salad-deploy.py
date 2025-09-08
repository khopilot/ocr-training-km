#!/usr/bin/env python3
"""
SaladCloud Deployment Script for Khmer OCR Training

This script deploys the Khmer OCR training pipeline to SaladCloud
using their API with 8x L40S GPUs for distributed training.
"""

import json
import os
import time
import requests
import base64
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

class SaladCloudDeployer:
    """SaladCloud API client for deploying training workloads"""
    
    def __init__(self, api_key: str, org_name: str = "khmer-ocr"):
        """Initialize SaladCloud deployer
        
        Args:
            api_key: SaladCloud API key
            org_name: Organization name
        """
        self.api_key = api_key
        self.org_name = org_name
        self.base_url = "https://api.salad.com/api/public"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
    def create_container_group(self, config: Dict[str, Any]) -> str:
        """Create a new container group
        
        Args:
            config: Container group configuration
            
        Returns:
            Container group ID
        """
        print(f"🚀 Creating container group: {config['name']}")
        
        # Prepare the container group specification
        spec = {
            "name": config["name"],
            "container": {
                "image": "nvidia/cuda:12.1.0-devel-ubuntu22.04",
                "command": ["/bin/bash"],
                "args": ["/app/deploy/salad/training-entrypoint.sh"],
                "resources": {
                    "cpu": config["resources"]["cpu"],
                    "memory": f"{config['resources']['memory']}Gi", 
                    "gpu_classes": [
                        {
                            "name": "rtx_4090_24gb", # Adjust for actual L40S class
                            "quantity": config["resources"]["gpu_count"]
                        }
                    ]
                },
                "environment_variables": self._format_env_vars(config.get("env_vars", {})),
                "working_directory": "/app"
            },
            "networking": {
                "protocol": "http",
                "port": 8080,
                "auth": False
            },
            "replicas": {
                "min": 1,
                "max": 1,
                "desired": 1
            },
            "restart_policy": {
                "restart_policy": "never",
                "max_restarts": 0
            },
            "probes": {
                "startup_probe": {
                    "http": {
                        "path": "/health",
                        "port": 8080
                    },
                    "initial_delay_seconds": 300,
                    "timeout_seconds": 30,
                    "period_seconds": 10,
                    "failure_threshold": 30
                },
                "liveness_probe": {
                    "http": {
                        "path": "/health", 
                        "port": 8080
                    },
                    "initial_delay_seconds": 600,
                    "timeout_seconds": 30,
                    "period_seconds": 60,
                    "failure_threshold": 3
                }
            },
            "country_codes": ["us", "ca"],
            "auto_start_policy": True
        }
        
        # Create the container group
        url = f"{self.base_url}/organizations/{self.org_name}/projects/{config['project']}/container-groups"
        
        response = self.session.post(url, json=spec)
        
        if response.status_code == 201:
            container_group = response.json()
            print(f"✅ Container group created: {container_group['id']}")
            return container_group['id']
        else:
            print(f"❌ Failed to create container group: {response.status_code}")
            print(f"Response: {response.text}")
            raise Exception(f"Container group creation failed: {response.text}")
    
    def _format_env_vars(self, env_vars: Dict[str, str]) -> list:
        """Format environment variables for SaladCloud API"""
        return [{"name": k, "value": str(v)} for k, v in env_vars.items()]
    
    def start_container_group(self, project_name: str, container_group_id: str) -> bool:
        """Start a container group
        
        Args:
            project_name: Project name
            container_group_id: Container group ID
            
        Returns:
            True if successful
        """
        print(f"▶️  Starting container group: {container_group_id}")
        
        url = f"{self.base_url}/organizations/{self.org_name}/projects/{project_name}/container-groups/{container_group_id}/start"
        
        response = self.session.post(url)
        
        if response.status_code == 202:
            print("✅ Container group start initiated")
            return True
        else:
            print(f"❌ Failed to start container group: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    def get_container_group_status(self, project_name: str, container_group_id: str) -> Dict[str, Any]:
        """Get container group status
        
        Args:
            project_name: Project name  
            container_group_id: Container group ID
            
        Returns:
            Status information
        """
        url = f"{self.base_url}/organizations/{self.org_name}/projects/{project_name}/container-groups/{container_group_id}"
        
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return {}
    
    def monitor_training(self, project_name: str, container_group_id: str, 
                        max_hours: int = 8) -> bool:
        """Monitor training progress
        
        Args:
            project_name: Project name
            container_group_id: Container group ID
            max_hours: Maximum training time in hours
            
        Returns:
            True if training completed successfully
        """
        start_time = time.time()
        max_seconds = max_hours * 3600
        
        print(f"👀 Monitoring training progress (max {max_hours} hours)...")
        
        while time.time() - start_time < max_seconds:
            status = self.get_container_group_status(project_name, container_group_id)
            
            if not status:
                time.sleep(60)
                continue
                
            current_state = status.get('current_state', {})
            status_text = current_state.get('status', 'unknown')
            
            elapsed_hours = (time.time() - start_time) / 3600
            print(f"📊 Status: {status_text} | Elapsed: {elapsed_hours:.1f}h | Cost: ${elapsed_hours * 2.56:.2f}")
            
            # Check if training is completed
            if status_text == 'stopped':
                print("🎉 Training completed!")
                return True
            elif status_text == 'failed':
                print("❌ Training failed!")
                return False
                
            # Wait before next check
            time.sleep(180)  # Check every 3 minutes
        
        print("⏰ Training timeout reached")
        return False
    
    def stop_container_group(self, project_name: str, container_group_id: str) -> bool:
        """Stop a container group
        
        Args:
            project_name: Project name
            container_group_id: Container group ID
            
        Returns:
            True if successful
        """
        print(f"⏹️  Stopping container group: {container_group_id}")
        
        url = f"{self.base_url}/organizations/{self.org_name}/projects/{project_name}/container-groups/{container_group_id}/stop"
        
        response = self.session.post(url)
        
        if response.status_code == 202:
            print("✅ Container group stop initiated")
            return True
        else:
            print(f"❌ Failed to stop container group: {response.status_code}")
            return False
    
    def delete_container_group(self, project_name: str, container_group_id: str) -> bool:
        """Delete a container group
        
        Args:
            project_name: Project name
            container_group_id: Container group ID
            
        Returns:
            True if successful
        """
        print(f"🗑️  Deleting container group: {container_group_id}")
        
        url = f"{self.base_url}/organizations/{self.org_name}/projects/{project_name}/container-groups/{container_group_id}"
        
        response = self.session.delete(url)
        
        if response.status_code == 204:
            print("✅ Container group deleted")
            return True
        else:
            print(f"❌ Failed to delete container group: {response.status_code}")
            return False


def load_config() -> Dict[str, Any]:
    """Load configuration from environment and files"""
    config = {
        "name": "khmer-ocr-training",
        "project": "khmer-ocr-gpu-training",
        "resources": {
            "cpu": 128,
            "memory": 512,
            "gpu_count": 8
        },
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "SERVICE_VARIANT": "paddle",
            "PRODUCTION_MODE": "prod",
            "USE_GPU": "true",
            "NUM_GPUS": "8",
            "BATCH_SIZE_PER_GPU": "16",
            "TOTAL_BATCH_SIZE": "128",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "WEBHOOK_SECRET_KEY": os.getenv("WEBHOOK_SECRET_KEY", ""),
            "MODEL_OUTPUT_URL": os.getenv("MODEL_OUTPUT_URL", "https://webhook.site/test")
        }
    }
    
    return config


def main():
    """Main deployment function"""
    print("🥗 SaladCloud Khmer OCR Training Deployment")
    print("=" * 50)
    
    # Load environment variables
    api_key = os.getenv("SALAD_API_KEY")
    if not api_key:
        print("❌ SALAD_API_KEY environment variable required")
        return 1
    
    webhook_secret = os.getenv("WEBHOOK_SECRET_KEY")
    if not webhook_secret:
        print("❌ WEBHOOK_SECRET_KEY environment variable required")
        return 1
        
    # Initialize deployer
    deployer = SaladCloudDeployer(api_key)
    
    # Load configuration
    config = load_config()
    
    try:
        # Create container group
        container_group_id = deployer.create_container_group(config)
        
        # Start training
        if deployer.start_container_group(config["project"], container_group_id):
            print(f"🚀 Training started! Container Group ID: {container_group_id}")
            print(f"💰 Estimated cost: ~${2.56 * 6:.2f} for 6-hour training")
            print(f"📊 Monitor at: https://portal.salad.com")
            
            # Monitor training progress
            success = deployer.monitor_training(config["project"], container_group_id, max_hours=8)
            
            if success:
                print("🎉 Training completed successfully!")
                print("📥 Check your webhook endpoint for model downloads")
            else:
                print("❌ Training did not complete successfully")
                
        else:
            print("❌ Failed to start training")
            return 1
            
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return 1
    
    # Cleanup option
    cleanup = input("\n🧹 Delete container group? (y/N): ").strip().lower()
    if cleanup == 'y':
        deployer.delete_container_group(config["project"], container_group_id)
    else:
        print(f"📋 Container Group ID: {container_group_id}")
        print("💡 Use this ID to manage the deployment manually")
    
    return 0


if __name__ == "__main__":
    exit(main())