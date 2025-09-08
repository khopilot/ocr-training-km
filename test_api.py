
import subprocess
import time
import sys

# Start server
proc = subprocess.Popen(
    [sys.executable, "service/app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for startup
time.sleep(5)

# Check if running
if proc.poll() is None:
    print("✅ API server started on http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    
    # Test health endpoint
    import requests
    try:
        resp = requests.get("http://localhost:8000/health", timeout=5)
        if resp.status_code == 200:
            print("✅ Health check passed")
    except:
        print("⚠️ Health check failed")
    
    # Kill server
    proc.terminate()
    print("Server stopped (was just a test)")
else:
    print("❌ Server failed to start")
