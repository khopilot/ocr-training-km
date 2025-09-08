#!/usr/bin/env python3
"""Integration tests for service startup with different backends"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


class TestServiceStartup(unittest.TestCase):
    """Test service startup with different backend configurations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name) / "models"
        self.models_dir.mkdir()
        
        # Mock external dependencies
        self.paddle_mock = MagicMock()
        self.onnx_mock = MagicMock()
        self.tesseract_mock = MagicMock()
        
        # Set up module patches
        self.module_patches = [
            patch('paddleocr.PaddleOCR', self.paddle_mock),
            patch('onnxruntime.InferenceSession', self.onnx_mock),
            patch('pytesseract.get_tesseract_version', self.tesseract_mock),
        ]
        
        for p in self.module_patches:
            p.start()
    
    def tearDown(self):
        """Clean up test fixtures"""
        for p in self.module_patches:
            p.stop()
        self.temp_dir.cleanup()
    
    def test_paddle_backend_demo_mode(self):
        """Test PaddleOCR backend in demo mode"""
        test_env = {
            'SERVICE_VARIANT': 'paddle',
            'PRODUCTION_MODE': 'demo',
            'PADDLE_LANG': 'ch'
        }
        
        with patch.dict(os.environ, test_env):
            with patch('service.app._paddle_available', return_value=True):
                with patch('service.app.Path') as mock_path:
                    mock_path.return_value = self.models_dir
                    
                    # Mock the service startup
                    with patch('service.app.load_models') as mock_load:
                        mock_load.return_value = None
                        
                        # Import and reload to get fresh environment
                        import importlib
                        import service.app
                        importlib.reload(service.app)
                        
                        # Call load_models to test backend selection
                        service.app.load_models()
                        
                        # Verify PaddleOCR was initialized
                        self.paddle_mock.assert_called_once()
                        call_args = self.paddle_mock.call_args
                        self.assertTrue(call_args.kwargs['use_angle_cls'])
                        self.assertEqual(call_args.kwargs['lang'], 'ch')
                        self.assertIsNone(call_args.kwargs.get('det_model_dir'))
                        self.assertIsNone(call_args.kwargs.get('rec_model_dir'))
    
    def test_paddle_backend_production_mode(self):
        """Test PaddleOCR backend in production mode with trained models"""
        # Create mock trained model directories
        dbnet_dir = self.models_dir / "dbnet_trained"
        dbnet_dir.mkdir()
        (dbnet_dir / "inference.pdmodel").touch()
        
        rec_dir = self.models_dir / "rec_kh_trained" 
        rec_dir.mkdir()
        (rec_dir / "inference.pdmodel").touch()
        
        test_env = {
            'SERVICE_VARIANT': 'paddle',
            'PRODUCTION_MODE': 'prod',
            'PADDLE_LANG': 'auto'
        }
        
        with patch.dict(os.environ, test_env):
            with patch('service.app._paddle_available', return_value=True):
                with patch('service.app._get_model_path') as mock_get_path:
                    # Mock model path detection
                    def mock_path_side_effect(model_type):
                        if model_type == 'det':
                            return str(dbnet_dir)
                        elif model_type == 'rec':
                            return str(rec_dir)
                        return None
                    
                    mock_get_path.side_effect = mock_path_side_effect
                    
                    with patch('service.app.Path') as mock_path_cls:
                        mock_path_cls.return_value.exists.return_value = True
                        
                        # Import and reload to get fresh environment
                        import importlib
                        import service.app
                        importlib.reload(service.app)
                        
                        # Call load_models to test production mode
                        service.app.load_models()
                        
                        # Verify PaddleOCR was initialized with trained models
                        self.paddle_mock.assert_called_once()
                        call_args = self.paddle_mock.call_args
                        self.assertEqual(call_args.kwargs['det_model_dir'], str(dbnet_dir))
                        self.assertEqual(call_args.kwargs['rec_model_dir'], str(rec_dir))
    
    def test_onnx_backend_fallback(self):
        """Test ONNX backend when PaddleOCR unavailable"""
        test_env = {
            'SERVICE_VARIANT': 'auto',
            'PRODUCTION_MODE': 'demo'
        }
        
        with patch.dict(os.environ, test_env):
            with patch('service.app._paddle_available', return_value=False):
                with patch('service.app._onnx_available', return_value=True):
                    with patch('onnxruntime.InferenceSession') as mock_onnx:
                        
                        # Import and reload to get fresh environment
                        import importlib
                        import service.app
                        importlib.reload(service.app)
                        
                        # Call load_models to test ONNX fallback
                        service.app.load_models()
                        
                        # Verify service selected ONNX backend
                        self.assertEqual(service.app.SERVICE_VARIANT, "onnx")
                        self.assertIsNotNone(service.app.BACKEND_ENGINE)
                        self.assertEqual(service.app.BACKEND_ENGINE["type"], "onnx")
    
    def test_tesseract_fallback(self):
        """Test Tesseract fallback when other backends unavailable"""
        test_env = {
            'SERVICE_VARIANT': 'auto',
            'PRODUCTION_MODE': 'demo'
        }
        
        with patch.dict(os.environ, test_env):
            with patch('service.app._paddle_available', return_value=False):
                with patch('service.app._onnx_available', return_value=False):
                    with patch('pytesseract.get_tesseract_version', return_value="5.0.0"):
                        
                        # Import and reload to get fresh environment
                        import importlib
                        import service.app
                        importlib.reload(service.app)
                        
                        # Call load_models to test Tesseract fallback
                        service.app.load_models()
                        
                        # Verify service selected Tesseract backend
                        self.assertEqual(service.app.SERVICE_VARIANT, "tesseract")
                        self.assertIsNotNone(service.app.BACKEND_ENGINE)
                        self.assertEqual(service.app.BACKEND_ENGINE["type"], "tesseract")
    
    def test_forced_backend_selection(self):
        """Test forced backend selection bypasses auto-detection"""
        test_env = {
            'SERVICE_VARIANT': 'tesseract',
            'PRODUCTION_MODE': 'demo'
        }
        
        with patch.dict(os.environ, test_env):
            with patch('service.app._paddle_available', return_value=True):  # PaddleOCR available
                with patch('pytesseract.get_tesseract_version', return_value="5.0.0"):
                    
                    # Import and reload to get fresh environment
                    import importlib
                    import service.app
                    importlib.reload(service.app)
                    
                    # Call load_models with forced Tesseract
                    service.app.load_models()
                    
                    # Verify Tesseract was selected despite PaddleOCR availability
                    self.assertEqual(service.app.SERVICE_VARIANT, "tesseract")
                    self.paddle_mock.assert_not_called()
    
    def test_backend_failure_fallback(self):
        """Test backend failure triggers fallback to next option"""
        test_env = {
            'SERVICE_VARIANT': 'auto',
            'PRODUCTION_MODE': 'demo'
        }
        
        with patch.dict(os.environ, test_env):
            with patch('service.app._paddle_available', return_value=True):
                with patch('service.app._onnx_available', return_value=True):
                    # Make PaddleOCR fail during initialization
                    self.paddle_mock.side_effect = Exception("PaddleOCR init failed")
                    
                    # Import and reload to get fresh environment
                    import importlib
                    import service.app
                    importlib.reload(service.app)
                    
                    # Call load_models - should fallback to ONNX
                    service.app.load_models()
                    
                    # Verify service fell back to ONNX
                    self.assertEqual(service.app.SERVICE_VARIANT, "onnx")
                    self.assertIsNotNone(service.app.BACKEND_ENGINE)
    
    def test_no_backends_available(self):
        """Test error when no backends are available"""
        test_env = {
            'SERVICE_VARIANT': 'auto',
            'PRODUCTION_MODE': 'demo'
        }
        
        with patch.dict(os.environ, test_env):
            with patch('service.app._paddle_available', return_value=False):
                with patch('service.app._onnx_available', return_value=False):
                    with patch('pytesseract.get_tesseract_version', side_effect=Exception("No tesseract")):
                        
                        # Import and reload to get fresh environment
                        import importlib
                        import service.app
                        importlib.reload(service.app)
                        
                        # Should raise RuntimeError when no backends available
                        with self.assertRaises(RuntimeError) as context:
                            service.app.load_models()
                        
                        self.assertIn("No OCR backend available", str(context.exception))


class TestServiceMetrics(unittest.TestCase):
    """Test service startup logging and metrics"""
    
    @patch('service.app.print')
    def test_startup_logging_demo_mode(self, mock_print):
        """Test startup logging in demo mode"""
        test_env = {
            'SERVICE_VARIANT': 'tesseract',
            'PRODUCTION_MODE': 'demo'
        }
        
        with patch.dict(os.environ, test_env):
            with patch('pytesseract.get_tesseract_version', return_value="5.0.0"):
                
                # Import and reload to get fresh environment
                import importlib
                import service.app
                importlib.reload(service.app)
                
                # Call load_models to trigger logging
                service.app.load_models()
                
                # Verify startup messages were printed
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                startup_messages = [msg for msg in print_calls if "Loading OCR backend" in msg]
                success_messages = [msg for msg in print_calls if "OCR service ready" in msg]
                
                self.assertTrue(any("tesseract" in msg and "demo" in msg for msg in startup_messages))
                self.assertTrue(any("tesseract" in msg and "demo" in msg for msg in success_messages))
    
    @patch('service.app.print')
    def test_startup_logging_production_mode(self, mock_print):
        """Test startup logging in production mode"""
        test_env = {
            'SERVICE_VARIANT': 'paddle',
            'PRODUCTION_MODE': 'prod'
        }
        
        with patch.dict(os.environ, test_env):
            with patch('service.app._paddle_available', return_value=True):
                with patch('paddleocr.PaddleOCR'):
                    
                    # Import and reload to get fresh environment
                    import importlib
                    import service.app
                    importlib.reload(service.app)
                    
                    # Call load_models to trigger logging
                    service.app.load_models()
                    
                    # Verify production mode logging
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    config_messages = [msg for msg in print_calls if "Production mode" in msg]
                    
                    self.assertTrue(any("trained Khmer models" in msg for msg in config_messages))


if __name__ == '__main__':
    unittest.main()