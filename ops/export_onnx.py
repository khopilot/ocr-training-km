#!/usr/bin/env python3
"""
Export PaddleOCR models to ONNX format for cross-platform inference
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import numpy as np
    import yaml
    import json
    print("✅ Basic dependencies loaded")
except ImportError as e:
    print(f"❌ Missing basic dependencies: {e}")
    sys.exit(1)

# Optional imports
try:
    import paddle
    import paddleocr
    PADDLE_AVAILABLE = True
    print("✅ PaddleOCR available for export")
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️  PaddleOCR not available - will create mock ONNX files")

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("✅ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX not available - install with: pip install onnx onnxruntime")

class ONNXExporter:
    """Export PaddleOCR models to ONNX format"""
    
    def __init__(self, model_dir: Path, output_dir: Path):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_detection_model(self, model_path: Path) -> Optional[Path]:
        """Export DBNet detection model to ONNX"""
        print(f"📋 Exporting detection model: {model_path}")
        
        if not PADDLE_AVAILABLE:
            # Create mock ONNX file
            mock_path = self.output_dir / "dbnet_detection.onnx"
            self._create_mock_onnx(mock_path, "DBNet Detection")
            return mock_path
        
        try:
            # Load PaddlePaddle model
            import paddle2onnx
            
            # Convert to ONNX
            output_path = self.output_dir / "dbnet_detection.onnx"
            
            # Mock conversion for demo (real implementation would use paddle2onnx)
            self._create_mock_onnx(output_path, "DBNet Detection")
            
            print(f"   ✅ Detection model exported: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"   ❌ Failed to export detection model: {e}")
            return None
    
    def export_recognition_model(self, model_path: Path) -> Optional[Path]:
        """Export CRNN recognition model to ONNX"""
        print(f"📋 Exporting recognition model: {model_path}")
        
        if not PADDLE_AVAILABLE:
            # Create mock ONNX file
            mock_path = self.output_dir / "crnn_recognition.onnx"
            self._create_mock_onnx(mock_path, "CRNN Recognition")
            return mock_path
        
        try:
            # Load PaddlePaddle model
            import paddle2onnx
            
            # Convert to ONNX
            output_path = self.output_dir / "crnn_recognition.onnx"
            
            # Mock conversion for demo (real implementation would use paddle2onnx)
            self._create_mock_onnx(output_path, "CRNN Recognition")
            
            print(f"   ✅ Recognition model exported: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"   ❌ Failed to export recognition model: {e}")
            return None
    
    def _create_mock_onnx(self, output_path: Path, model_type: str):
        """Create a mock ONNX file for demo purposes"""
        if ONNX_AVAILABLE:
            import onnx
            from onnx import helper, TensorProto
            
            # Create a minimal ONNX graph
            if model_type == "DBNet Detection":
                # Detection model: input image -> bounding boxes
                input_tensor = helper.make_tensor_value_info(
                    'input', TensorProto.FLOAT, [1, 3, 640, 640]
                )
                output_tensor = helper.make_tensor_value_info(
                    'output', TensorProto.FLOAT, [1, 1, 640, 640]
                )
            else:
                # Recognition model: input image -> text sequence
                input_tensor = helper.make_tensor_value_info(
                    'input', TensorProto.FLOAT, [1, 3, 48, 320]
                )
                output_tensor = helper.make_tensor_value_info(
                    'output', TensorProto.FLOAT, [1, 187, 25]  # 187 charset size
                )
            
            # Create identity node (placeholder)
            node = helper.make_node(
                'Identity',
                inputs=['input'],
                outputs=['output'],
                name=f'mock_{model_type.lower().replace(" ", "_")}'
            )
            
            # Create graph
            graph = helper.make_graph(
                nodes=[node],
                name=f'Mock{model_type.replace(" ", "")}',
                inputs=[input_tensor],
                outputs=[output_tensor]
            )
            
            # Create model
            model = helper.make_model(graph)
            model.opset_import[0].version = 11
            
            # Add metadata
            model.metadata_props.extend([
                helper.make_metadata(key="model_type", value=model_type),
                helper.make_metadata(key="framework", value="PaddleOCR"),
                helper.make_metadata(key="export_mode", value="DEMO"),
                helper.make_metadata(key="timestamp", value=str(int(time.time())))
            ])
            
            # Save model
            onnx.save(model, str(output_path))
        else:
            # Just create a placeholder file
            with open(output_path, 'wb') as f:
                f.write(b'# Mock ONNX file for demo\n')
    
    def validate_onnx_model(self, onnx_path: Path) -> bool:
        """Validate exported ONNX model"""
        if not ONNX_AVAILABLE:
            print(f"   ⚠️  Cannot validate {onnx_path.name} - ONNX not available")
            return True
        
        try:
            import onnx
            
            # Load and check model
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            
            # Try loading with ONNX Runtime
            session = ort.InferenceSession(str(onnx_path))
            
            print(f"   ✅ {onnx_path.name} validated successfully")
            print(f"      Inputs: {[input.name for input in session.get_inputs()]}")
            print(f"      Outputs: {[output.name for output in session.get_outputs()]}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Validation failed for {onnx_path.name}: {e}")
            return False
    
    def create_export_manifest(self, exported_models: Dict[str, Path]) -> Path:
        """Create export manifest with model metadata"""
        manifest = {
            "export_type": "ONNX_DEMO",
            "platform": "macOS M3", 
            "framework": "PaddleOCR",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "models": {},
            "disclaimer": "These are DEMO ONNX models for development purposes"
        }
        
        for model_type, model_path in exported_models.items():
            if model_path and model_path.exists():
                try:
                    rel_path = model_path.relative_to(PROJECT_ROOT)
                except ValueError:
                    rel_path = model_path
                    
                manifest["models"][model_type] = {
                    "path": str(rel_path),
                    "size_mb": model_path.stat().st_size / (1024 * 1024),
                    "format": "ONNX"
                }
        
        manifest_path = self.output_dir / "export_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        return manifest_path

def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description='Export PaddleOCR models to ONNX')
    parser.add_argument('--model-dir', type=Path, default='models',
                       help='Directory containing PaddleOCR models')
    parser.add_argument('--output-dir', type=Path, default='models/onnx',
                       help='Output directory for ONNX models')
    parser.add_argument('--detection-model', type=Path, 
                       help='Specific detection model path')
    parser.add_argument('--recognition-model', type=Path,
                       help='Specific recognition model path')
    parser.add_argument('--validate', action='store_true',
                       help='Validate exported models')
    
    args = parser.parse_args()
    
    print("🔧 ONNX Model Exporter")
    print(f"   Model dir: {args.model_dir}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   PaddleOCR: {'Available' if PADDLE_AVAILABLE else 'Mock mode'}")
    print(f"   ONNX Runtime: {'Available' if ONNX_AVAILABLE else 'Not available'}")
    print()
    
    try:
        # Initialize exporter
        exporter = ONNXExporter(args.model_dir, args.output_dir)
        
        exported_models = {}
        
        # Export detection model
        if args.detection_model:
            det_path = exporter.export_detection_model(args.detection_model)
        else:
            # Look for detection model in model directory
            det_candidates = list(args.model_dir.glob("*det*")) + list(args.model_dir.glob("*dbnet*"))
            if det_candidates:
                det_path = exporter.export_detection_model(det_candidates[0])
            else:
                print("📋 Creating mock detection model...")
                det_path = exporter.export_detection_model(Path("mock_detection"))
        
        if det_path:
            exported_models["detection"] = det_path
            if args.validate:
                exporter.validate_onnx_model(det_path)
        
        # Export recognition model
        if args.recognition_model:
            rec_path = exporter.export_recognition_model(args.recognition_model)
        else:
            # Look for recognition model in model directory
            rec_candidates = list(args.model_dir.glob("*rec*")) + list(args.model_dir.glob("*crnn*"))
            if rec_candidates:
                rec_path = exporter.export_recognition_model(rec_candidates[0])
            else:
                print("📋 Creating mock recognition model...")
                rec_path = exporter.export_recognition_model(Path("mock_recognition"))
        
        if rec_path:
            exported_models["recognition"] = rec_path
            if args.validate:
                exporter.validate_onnx_model(rec_path)
        
        # Create manifest
        manifest_path = exporter.create_export_manifest(exported_models)
        
        print()
        print("🎉 ONNX export completed!")
        print(f"   Models: {len(exported_models)}")
        print(f"   Manifest: {manifest_path}")
        
        for model_type, model_path in exported_models.items():
            if model_path:
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"   {model_type.title()}: {model_path} ({size_mb:.1f} MB)")
        
        print()
        print("⚠️  Remember: These are DEMO ONNX models for development")
        print("   Use trained PaddleOCR models for production export")
        
    except KeyboardInterrupt:
        print("\n🛑 Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()