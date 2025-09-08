#!/usr/bin/env python3
"""Pre-edit hook for Khmer OCR project - validates code before edits"""

import sys
import json
import re
from pathlib import Path

def validate_edit(file_path: str, content: str) -> dict:
    """Validate edits before applying"""
    warnings = []
    errors = []
    
    # Check for credentials or secrets
    secret_patterns = [
        r'["\']?(api[_-]?key|secret|password|token|credentials)["\']?\s*[:=]\s*["\'][^"\']+["\']',
        r'(AWS|AZURE|GCP|OPENAI)_[A-Z_]+\s*=\s*["\'][^"\']+["\']',
        r'(sk-|pk-|api-)[a-zA-Z0-9]{32,}'
    ]
    
    for pattern in secret_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            errors.append(f"Potential secret/credential detected: {pattern}")
    
    # Check for model file paths
    if file_path.endswith('.py'):
        # Ensure model paths use manifest
        if 'paddle' in content.lower() or 'model' in content.lower():
            if '/models/' in content and 'manifest.json' not in content:
                warnings.append("Model paths should reference manifest.json for version control")
        
        # Check for hardcoded versions
        if re.search(r'paddleocr==\d+\.\d+\.\d+', content):
            warnings.append("Dependencies should be in pyproject.toml, not hardcoded")
    
    # Check for proper error handling in service files
    if '/service/' in file_path and file_path.endswith('.py'):
        if 'try:' in content and 'except Exception' in content:
            warnings.append("Avoid bare 'except Exception' - use specific exceptions")
    
    # Validate schema changes
    if file_path.endswith('schemas.py'):
        if 'BaseModel' in content:
            warnings.append("Remember to update manifest.json version after schema changes")
    
    return {
        "status": "error" if errors else "warning" if warnings else "ok",
        "errors": errors,
        "warnings": warnings
    }

if __name__ == "__main__":
    # Get input from Claude Code
    file_path = sys.argv[1] if len(sys.argv) > 1 else ""
    content = sys.stdin.read() if not sys.stdin.isatty() else ""
    
    result = validate_edit(file_path, content)
    
    # Output result
    print(json.dumps(result, indent=2))
    
    # Exit with error if validation failed
    if result["status"] == "error":
        sys.exit(1)