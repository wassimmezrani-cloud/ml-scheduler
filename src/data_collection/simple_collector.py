#!/usr/bin/env python3
"""
Premier collecteur de données pour ML Scheduler HYDATIS
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class SimplePrometheusCollector:
    def __init__(self, prometheus_url="http://10.110.190.83:9090"):
        self.prometheus_url = prometheus_url
    
    def collect_node_metrics_now(self):
        """Collecte instantanée métriques nodes"""
        
        queries = {
            'cpu_usage': 'rate(node_cpu_seconds_total{mode!="idle"}[5m])',
            'memory_usage': '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes',
            'load_1m': 'node_load1'
        }
        
        results = {}
        
        for metric_name, query in queries.items():
            url = f"{self.prometheus_url}/api/v1/query"
            params = {'query': query}
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'success':
                        results[metric_name] = data['data']['result']
                        print(f"✅ {metric_name}: {len(data['data']['result'])} samples")
                    else:
                        print(f"❌ {metric_name}: query failed")
                else:
                    print(f"❌ {metric_name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {metric_name}: {e}")
        
        return results
    
    def save_initial_dataset(self, metrics):
        """Sauvegarder dataset initial"""
        
        os.makedirs('data/raw', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/raw/initial_metrics_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"✅ Données sauvegardées: {filename}")
        return filename

def main():
    collector = SimplePrometheusCollector()
    
    print("🚀 HYDATIS ML Scheduler - Première Collecte")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test collecte
    metrics = collector.collect_node_metrics_now()
    
    if metrics:
        filename = collector.save_initial_dataset(metrics)
        print(f"\n✅ SUCCESS: Première collecte réussie")
        print(f"📊 Métriques: {list(metrics.keys())}")
        print(f"💾 Fichier: {filename}")
        
        return True
    else:
        print("\n❌ ÉCHEC: Aucune donnée collectée")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
