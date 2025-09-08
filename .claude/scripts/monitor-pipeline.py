#!/usr/bin/env python3
"""Monitor Khmer OCR pipeline execution and collect metrics"""

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
import subprocess
import psutil
import sys

class PipelineMonitor:
    def __init__(self, project_root="/Users/niko/Desktop/khmer-ocr-v1"):
        self.project_root = Path(project_root)
        self.metrics_db = Path.home() / ".claude" / "metrics.db"
        self.log_file = self.project_root / ".claude" / "logs" / "pipeline.log"
        self.init_db()
        
    def init_db(self):
        """Initialize metrics database"""
        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                project TEXT,
                stage TEXT,
                metric_name TEXT,
                metric_value REAL,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE,
                start_time TEXT,
                end_time TEXT,
                status TEXT,
                config TEXT,
                metrics TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_metric(self, stage: str, metric_name: str, value: float, metadata: dict = None):
        """Log a metric to the database"""
        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO pipeline_metrics (timestamp, project, stage, metric_name, metric_value, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            "khmer-ocr-v1",
            stage,
            metric_name,
            value,
            json.dumps(metadata or {})
        ))
        
        conn.commit()
        conn.close()
        
        print(f"📊 {stage}/{metric_name}: {value}")
    
    def monitor_training(self, run_id: str):
        """Monitor a training run"""
        print(f"🔍 Monitoring training run: {run_id}")
        
        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()
        
        # Record run start
        cursor.execute("""
            INSERT INTO training_runs (run_id, start_time, status, config)
            VALUES (?, ?, ?, ?)
        """, (run_id, datetime.now().isoformat(), "running", "{}"))
        conn.commit()
        
        # Monitor system resources
        start_time = time.time()
        while True:
            try:
                # CPU and Memory usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.log_metric("training", "cpu_usage", cpu_percent, {"run_id": run_id})
                self.log_metric("training", "memory_usage", memory.percent, {"run_id": run_id})
                
                # GPU usage (if nvidia-smi available)
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        gpu_util, mem_used, mem_total = result.stdout.strip().split(", ")
                        self.log_metric("training", "gpu_usage", float(gpu_util), {"run_id": run_id})
                        self.log_metric("training", "gpu_memory", float(mem_used) / float(mem_total) * 100, 
                                      {"run_id": run_id})
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                
                # Check for training completion
                if self.check_training_complete(run_id):
                    break
                    
                # Check every 30 seconds
                time.sleep(30)
                
                # Timeout after 8 hours
                if time.time() - start_time > 8 * 3600:
                    print("⚠️ Training timeout reached")
                    break
                    
            except KeyboardInterrupt:
                print("\n⏹️ Monitoring stopped by user")
                break
        
        # Record run end
        cursor.execute("""
            UPDATE training_runs 
            SET end_time = ?, status = ?
            WHERE run_id = ?
        """, (datetime.now().isoformat(), "completed", run_id))
        conn.commit()
        conn.close()
        
        print(f"✅ Monitoring completed for run: {run_id}")
    
    def check_training_complete(self, run_id: str) -> bool:
        """Check if training is complete"""
        # Check for completion markers
        markers = [
            self.project_root / "models" / "dbnet_best.pdparams",
            self.project_root / "models" / "rec_kh_best.pdparams",
            self.project_root / "eval" / "report.json"
        ]
        
        return any(marker.exists() for marker in markers)
    
    def generate_report(self):
        """Generate monitoring report"""
        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()
        
        # Get recent metrics
        cursor.execute("""
            SELECT stage, metric_name, AVG(metric_value) as avg_value, 
                   MAX(metric_value) as max_value, MIN(metric_value) as min_value
            FROM pipeline_metrics
            WHERE project = 'khmer-ocr-v1'
              AND timestamp > datetime('now', '-1 day')
            GROUP BY stage, metric_name
            ORDER BY stage, metric_name
        """)
        
        metrics = cursor.fetchall()
        
        print("\n📈 Pipeline Metrics Report (Last 24 hours)")
        print("=" * 60)
        
        current_stage = None
        for stage, metric, avg_val, max_val, min_val in metrics:
            if stage != current_stage:
                print(f"\n{stage.upper()}")
                current_stage = stage
            print(f"  {metric}: avg={avg_val:.2f}, max={max_val:.2f}, min={min_val:.2f}")
        
        # Get training runs
        cursor.execute("""
            SELECT run_id, start_time, end_time, status
            FROM training_runs
            WHERE start_time > datetime('now', '-7 days')
            ORDER BY start_time DESC
            LIMIT 5
        """)
        
        runs = cursor.fetchall()
        
        print("\n🏃 Recent Training Runs")
        print("-" * 60)
        for run_id, start_time, end_time, status in runs:
            duration = "N/A"
            if start_time and end_time:
                start = datetime.fromisoformat(start_time)
                end = datetime.fromisoformat(end_time)
                duration = str(end - start)
            print(f"  {run_id}: {status} (duration: {duration})")
        
        conn.close()

if __name__ == "__main__":
    monitor = PipelineMonitor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "train" and len(sys.argv) > 2:
            monitor.monitor_training(sys.argv[2])
        elif command == "report":
            monitor.generate_report()
        else:
            print("Usage: monitor-pipeline.py [train RUN_ID | report]")
    else:
        monitor.generate_report()