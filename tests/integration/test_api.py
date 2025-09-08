"""Integration tests for OCR API"""

import pytest
import requests
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import io


class TestOCRAPI:
    """Test OCR API endpoints"""
    
    BASE_URL = "http://localhost:8080"
    
    @classmethod
    def setup_class(cls):
        """Wait for service to be ready"""
        max_retries = 30
        for i in range(max_retries):
            try:
                r = requests.get(f"{cls.BASE_URL}/health")
                if r.status_code == 200:
                    return
            except:
                pass
            time.sleep(1)
        pytest.skip("Service not available")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "version" in data
        assert "uptime_seconds" in data
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = requests.get(f"{self.BASE_URL}/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check for key metrics
        metrics_text = response.text
        assert "ocr_requests_total" in metrics_text
        assert "ocr_request_duration_seconds" in metrics_text
        assert "ocr_errors_total" in metrics_text
    
    def test_ocr_endpoint_valid_image(self):
        """Test OCR endpoint with valid image"""
        # Create test image
        img = Image.new('RGB', (400, 100), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {'file': ('test.png', img_bytes, 'image/png')}
        data = {
            'enable_lm': True,
            'lm_weight': 0.3,
            'return_lines': True,
            'debug': False
        }
        
        response = requests.post(f"{self.BASE_URL}/ocr", files=files, data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "text" in result
        assert "lines" in result
        assert "version" in result
        assert "metrics" in result
        
        # Check metrics
        metrics = result["metrics"]
        assert "latency_ms" in metrics
        assert "request_id" in metrics
    
    def test_ocr_endpoint_invalid_file(self):
        """Test OCR endpoint with non-image file"""
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        
        response = requests.post(f"{self.BASE_URL}/ocr", files=files)
        assert response.status_code == 400
    
    def test_ocr_endpoint_with_request_id(self):
        """Test request ID tracking"""
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        request_id = "test-request-123"
        headers = {"X-Request-ID": request_id}
        files = {'file': ('test.png', img_bytes, 'image/png')}
        
        response = requests.post(
            f"{self.BASE_URL}/ocr", 
            files=files, 
            headers=headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["metrics"]["request_id"] == request_id
    
    def test_ocr_latency_threshold(self):
        """Test that latency meets thresholds"""
        # Create small test image for fast processing
        img = Image.new('RGB', (100, 50), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        latencies = []
        for _ in range(10):
            files = {'file': ('test.png', img_bytes, 'image/png')}
            img_bytes.seek(0)
            
            start = time.time()
            response = requests.post(f"{self.BASE_URL}/ocr", files=files)
            latency = (time.time() - start) * 1000
            
            assert response.status_code == 200
            latencies.append(latency)
        
        # Check P95 latency
        p95 = np.percentile(latencies, 95)
        # CPU threshold is more lenient
        assert p95 < 1000, f"P95 latency {p95:.2f}ms exceeds threshold"


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_full_pipeline(self):
        """Test complete OCR pipeline"""
        # This would test:
        # 1. Image upload
        # 2. Detection
        # 3. Recognition
        # 4. Language model rescoring
        # 5. Response formatting
        pass
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        # Would test service under load
        pass
    
    def test_large_image_handling(self):
        """Test handling of large images"""
        # Would test memory management
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])