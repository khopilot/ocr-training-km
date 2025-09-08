#!/usr/bin/env python3
"""Unit tests for service/app.py model detection and backend selection"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Mock the service dependencies to avoid import issues during testing
with patch.dict('sys.modules', {
    'paddleocr': MagicMock(),
    'onnxruntime': MagicMock(),
    'pytesseract': MagicMock(),
    'prometheus_client': MagicMock(),
}):
    from service.app import _get_model_path, _get_paddle_language, _paddle_available, _onnx_available


class TestModelPathDetection(unittest.TestCase):
    """Test model path detection functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name) / "models"
        self.models_dir.mkdir()
        
        # Patch the model directory in the function
        self.models_patch = patch('service.app.Path', return_value=self.models_dir)
        self.models_patch.start()
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.models_patch.stop()
        self.temp_dir.cleanup()
    
    def test_get_model_path_detection_models(self):
        """Test detection model path resolution"""
        # Create mock detection model files
        dbnet_dir = self.models_dir / "dbnet_trained"
        dbnet_dir.mkdir()
        (dbnet_dir / "dbnet_khmer.pdmodel").touch()
        (dbnet_dir / "dbnet_khmer.pdiparams").touch()
        
        with patch('service.app.Path') as mock_path:
            mock_path.return_value = self.models_dir
            
            # Test DBNet model detection
            result = _get_model_path('det')
            expected = str(dbnet_dir)
            
            # The function should find the dbnet model directory
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith("dbnet_trained"))
    
    def test_get_model_path_recognition_models(self):
        """Test recognition model path resolution"""
        # Create mock recognition model files
        rec_dir = self.models_dir / "rec_kh_trained"
        rec_dir.mkdir()
        (rec_dir / "rec_kh_khmer.pdmodel").touch()
        (rec_dir / "rec_kh_khmer.pdiparams").touch()
        
        with patch('service.app.Path') as mock_path:
            mock_path.return_value = self.models_dir
            
            # Test recognition model detection
            result = _get_model_path('rec')
            expected = str(rec_dir)
            
            # The function should find the recognition model directory
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith("rec_kh_trained"))
    
    def test_get_model_path_no_models(self):
        """Test behavior when no models are found"""
        with patch('service.app.Path') as mock_path:
            mock_path.return_value = self.models_dir
            
            # Test with no models present
            result_det = _get_model_path('det')
            result_rec = _get_model_path('rec')
            
            self.assertIsNone(result_det)
            self.assertIsNone(result_rec)
    
    def test_get_model_path_fallback_patterns(self):
        """Test fallback pattern matching"""
        # Create model with generic detection pattern
        det_dir = self.models_dir / "custom_detector"
        det_dir.mkdir()
        (det_dir / "text_det_v2.pdmodel").touch()
        
        # Create model with generic recognition pattern
        rec_dir = self.models_dir / "custom_recognizer"
        rec_dir.mkdir()
        (rec_dir / "text_rec_v2.pdmodel").touch()
        
        with patch('service.app.Path') as mock_path:
            mock_path.return_value = self.models_dir
            
            # Should find fallback patterns
            det_result = _get_model_path('det')
            rec_result = _get_model_path('rec')
            
            self.assertIsNotNone(det_result)
            self.assertIsNotNone(rec_result)
            self.assertTrue(det_result.endswith("custom_detector"))
            self.assertTrue(rec_result.endswith("custom_recognizer"))


class TestPaddleLanguageSelection(unittest.TestCase):
    """Test PaddleOCR language selection logic"""
    
    def setUp(self):
        """Set up environment variable mocks"""
        self.env_patcher = patch.dict(os.environ, {
            'PRODUCTION_MODE': 'demo',
            'PADDLE_LANG': 'auto'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up environment mocks"""
        self.env_patcher.stop()
    
    @patch('service.app.PADDLE_LANG', 'khmer')
    def test_explicit_paddle_lang(self):
        """Test explicit PADDLE_LANG override"""
        result = _get_paddle_language()
        self.assertEqual(result, 'khmer')
    
    @patch('service.app.PADDLE_LANG', 'auto')
    @patch('service.app.PRODUCTION_MODE', 'demo')
    def test_demo_mode_default(self):
        """Test demo mode defaults to multilingual"""
        result = _get_paddle_language()
        self.assertEqual(result, 'ch')  # Multilingual Chinese
    
    @patch('service.app.PADDLE_LANG', 'auto')
    @patch('service.app.PRODUCTION_MODE', 'prod')
    @patch('service.app._get_model_path')
    def test_production_mode_with_khmer_models(self, mock_get_path):
        """Test production mode with available Khmer models"""
        mock_get_path.return_value = "/models/rec_kh"
        
        with patch('service.app.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            result = _get_paddle_language()
            self.assertEqual(result, 'khmer')
    
    @patch('service.app.PADDLE_LANG', 'auto')
    @patch('service.app.PRODUCTION_MODE', 'prod')
    @patch('service.app._get_model_path')
    def test_production_mode_no_khmer_models(self, mock_get_path):
        """Test production mode fallback when no Khmer models"""
        mock_get_path.return_value = None
        
        result = _get_paddle_language()
        self.assertEqual(result, 'ch')  # Fallback to multilingual


class TestBackendAvailability(unittest.TestCase):
    """Test backend availability detection"""
    
    def test_paddle_available_success(self):
        """Test successful PaddleOCR detection"""
        with patch('service.app.paddleocr') as mock_paddle:
            result = _paddle_available()
            self.assertTrue(result)
    
    def test_paddle_available_import_error(self):
        """Test PaddleOCR import failure"""
        with patch('service.app.paddleocr', side_effect=ImportError):
            result = _paddle_available()
            self.assertFalse(result)
    
    def test_onnx_available_success(self):
        """Test successful ONNX Runtime detection"""
        with patch('service.app.onnxruntime') as mock_onnx:
            result = _onnx_available()
            self.assertTrue(result)
    
    def test_onnx_available_import_error(self):
        """Test ONNX Runtime import failure"""
        with patch('service.app.onnxruntime', side_effect=ImportError):
            result = _onnx_available()
            self.assertFalse(result)


class TestServiceConfiguration(unittest.TestCase):
    """Test service configuration integration"""
    
    def test_environment_variable_defaults(self):
        """Test that environment variables have correct defaults"""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to get fresh environment variable reads
            import importlib
            import service.app
            importlib.reload(service.app)
            
            self.assertEqual(service.app.SERVICE_VARIANT, "auto")
            self.assertEqual(service.app.PRODUCTION_MODE, "demo")
            self.assertEqual(service.app.PADDLE_LANG, "auto")
    
    def test_environment_variable_overrides(self):
        """Test environment variable overrides"""
        test_env = {
            'SERVICE_VARIANT': 'paddle',
            'PRODUCTION_MODE': 'prod', 
            'PADDLE_LANG': 'khmer'
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            # Re-import to get fresh environment variable reads
            import importlib
            import service.app
            importlib.reload(service.app)
            
            self.assertEqual(service.app.SERVICE_VARIANT, "paddle")
            self.assertEqual(service.app.PRODUCTION_MODE, "prod")
            self.assertEqual(service.app.PADDLE_LANG, "khmer")


if __name__ == '__main__':
    unittest.main()