# ========================================================================
# SMART CONTRACT SIMULATION AND PERFORMANCE ANALYSIS
# Blockchain-based OBD V2O System
# ========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import hashlib
import json
import time
import tracemalloc

# Set professional visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*70)
print("SMART CONTRACT SIMULATION AND PERFORMANCE ANALYSIS")
print("="*70)

# ========================================================================
# SMART CONTRACT IMPLEMENTATIONS
# ========================================================================

class AccessVerificationContract:
    """
    AVC: Authenticates entities requesting OBD data
    Verifies digital credentials and ownership signatures
    """
    def __init__(self):
        self.authorized_entities = {}
        self.access_logs = []
        
    def register_entity(self, entity_id, entity_type, public_key):
        """Register authorized entity"""
        self.authorized_entities[entity_id] = {
            'type': entity_type,
            'public_key': public_key,
            'registered_at': datetime.now().isoformat(),
            'access_count': 0
        }
        
    def verify_access(self, entity_id, signature, data_hash):
        """Verify entity access request"""
        start_time = time.time()
        
        if entity_id not in self.authorized_entities:
            return False, time.time() - start_time
        
        # Simulate signature verification
        entity = self.authorized_entities[entity_id]
        reconstructed = hashlib.sha256(
            f"{entity['public_key']}{data_hash}".encode()
        ).hexdigest()
        
        verified = signature == reconstructed[:32]
        
        if verified:
            entity['access_count'] += 1
            self.access_logs.append({
                'entity_id': entity_id,
                'timestamp': datetime.now().isoformat(),
                'verified': True
            })
        
        execution_time = time.time() - start_time
        return verified, execution_time


class DataIntegrityContract:
    """
    DIC: Monitors data authenticity
    Periodically compares stored and recalculated hash values
    """
    def __init__(self):
        self.data_registry = {}
        self.integrity_checks = []
        
    def register_data(self, data_id, data_hash, timestamp):
        """Register data with hash"""
        self.data_registry[data_id] = {
            'hash': data_hash,
            'timestamp': timestamp,
            'verified_count': 0
        }
        
    def verify_integrity(self, data_id, current_data):
        """Verify data integrity against stored hash"""
        start_time = time.time()
        
        if data_id not in self.data_registry:
            return False, time.time() - start_time
        
        stored_hash = self.data_registry[data_id]['hash']
        recalculated_hash = hashlib.sha256(
            json.dumps(current_data, sort_keys=True).encode()
        ).hexdigest()
        
        is_valid = stored_hash == recalculated_hash
        
        if is_valid:
            self.data_registry[data_id]['verified_count'] += 1
        
        self.integrity_checks.append({
            'data_id': data_id,
            'timestamp': datetime.now().isoformat(),
            'valid': is_valid
        })
        
        execution_time = time.time() - start_time
        return is_valid, execution_time


class ConsentManagementContract:
    """
    CMC: Enables dynamic access control
    Vehicle owner can grant, revoke, or update access privileges
    """
    def __init__(self, owner_id):
        self.owner_id = owner_id
        self.permissions = {}
        self.permission_logs = []
        
    def grant_permission(self, entity_id, entity_type, access_level, duration_days):
        """Grant access permission to entity"""
        start_time = time.time()
        
        self.permissions[entity_id] = {
            'type': entity_type,
            'access_level': access_level,
            'granted_at': datetime.now().isoformat(),
            'expires_at': (datetime.now().timestamp() + duration_days * 86400),
            'active': True
        }
        
        self.permission_logs.append({
            'action': 'grant',
            'entity_id': entity_id,
            'timestamp': datetime.now().isoformat()
        })
        
        execution_time = time.time() - start_time
        return True, execution_time
        
    def revoke_permission(self, entity_id):
        """Revoke access permission"""
        start_time = time.time()
        
        if entity_id in self.permissions:
            self.permissions[entity_id]['active'] = False
            self.permission_logs.append({
                'action': 'revoke',
                'entity_id': entity_id,
                'timestamp': datetime.now().isoformat()
            })
            result = True
        else:
            result = False
        
        execution_time = time.time() - start_time
        return result, execution_time
        
    def check_permission(self, entity_id):
        """Check if entity has valid permission"""
        start_time = time.time()
        
        if entity_id not in self.permissions:
            return False, time.time() - start_time
        
        perm = self.permissions[entity_id]
        is_valid = (perm['active'] and 
                   datetime.now().timestamp() < perm['expires_at'])
        
        execution_time = time.time() - start_time
        return is_valid, execution_time


# ========================================================================
# PERFORMANCE BENCHMARKING FUNCTIONS
# ========================================================================

def benchmark_contract_scalability(contract_class, operation_counts):
    """Benchmark contract performance at different scales"""
    throughputs = []
    latencies = []
    memory_usages = []
    
    for count in operation_counts:
        # Start memory tracking
        tracemalloc.start()
        
        if contract_class == 'AVC':
            contract = AccessVerificationContract()
            start = time.time()
            
            for i in range(count):
                contract.register_entity(f"entity_{i}", "mechanic", f"key_{i}")
                contract.verify_access(
                    f"entity_{i}", 
                    hashlib.sha256(f"key_{i}data_{i}".encode()).hexdigest()[:32],
                    f"data_{i}"
                )
            
            elapsed = time.time() - start
            
        elif contract_class == 'DIC':
            contract = DataIntegrityContract()
            start = time.time()
            
            for i in range(count):
                data = {'sensor': i, 'value': np.random.rand()}
                data_hash = hashlib.sha256(
                    json.dumps(data, sort_keys=True).encode()
                ).hexdigest()
                contract.register_data(f"data_{i}", data_hash, datetime.now())
                contract.verify_integrity(f"data_{i}", data)
            
            elapsed = time.time() - start
            
        elif contract_class == 'CMC':
            contract = ConsentManagementContract("owner_123")
            start = time.time()
            
            for i in range(count):
                contract.grant_permission(f"entity_{i}", "mechanic", "read", 30)
                contract.check_permission(f"entity_{i}")
            
            elapsed = time.time() - start
        
        # Memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        throughput = count / elapsed  # operations per second
        latency = (elapsed / count) * 1000  # milliseconds per operation
        memory_mb = peak / (1024 * 1024)  # MB
        
        throughputs.append(throughput)
        latencies.append(latency)
        memory_usages.append(memory_mb)
    
    return throughputs, latencies, memory_usages


# ========================================================================
# RUN BENCHMARKS
# ========================================================================

print("\n" + "="*70)
print("RUNNING PERFORMANCE BENCHMARKS")
print("="*70)

# Test at different scales
operation_counts = [100, 500, 1000, 2000, 5000, 10000]

contracts = {
    'Access Verification Contract (AVC)': 'AVC',
    'Data Integrity Contract (DIC)': 'DIC',
    'Consent Management Contract (CMC)': 'CMC'
}

results = {}

for name, code in contracts.items():
    print(f"\nBenchmarking {name}...")
    throughput, latency, memory = benchmark_contract_scalability(code, operation_counts)
    results[name] = {
        'throughput': throughput,
        'latency': latency,
        'memory': memory
    }
    print(f"✓ Completed: Max throughput = {max(throughput):.2f} ops/sec")

# ========================================================================
# VISUALIZATION 1: THROUGHPUT vs SCALE
# ========================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Smart Contract Throughput Performance', 
             fontsize=16, fontweight='bold')

for idx, (name, data) in enumerate(results.items()):
    ax = axes[idx]
    ax.plot(operation_counts, data['throughput'], 
           marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Operations', fontsize=10)
    ax.set_ylabel('Throughput (ops/sec)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

plt.tight_layout()
plt.savefig('smart_contract_throughput.png', dpi=300, bbox_inches='tight')
print("\n✓ Throughput graph saved: smart_contract_throughput.png")
plt.close()

# ========================================================================
# VISUALIZATION 2: LATENCY vs SCALE
# ========================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Smart Contract Latency Analysis', 
             fontsize=16, fontweight='bold')

for idx, (name, data) in enumerate(results.items()):
    ax = axes[idx]
    ax.plot(operation_counts, data['latency'], 
           marker='s', linewidth=2, markersize=8, color='darkorange')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Operations', fontsize=10)
    ax.set_ylabel('Latency (ms/operation)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

plt.tight_layout()
plt.savefig('smart_contract_latency.png', dpi=300, bbox_inches='tight')
print("✓ Latency graph saved: smart_contract_latency.png")
plt.close()

# ========================================================================
# VISUALIZATION 3: MEMORY USAGE vs SCALE
# ========================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Smart Contract Memory Consumption', 
             fontsize=16, fontweight='bold')

for idx, (name, data) in enumerate(results.items()):
    ax = axes[idx]
    ax.plot(operation_counts, data['memory'], 
           marker='^', linewidth=2, markersize=8, color='darkgreen')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Operations', fontsize=10)
    ax.set_ylabel('Memory Usage (MB)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

plt.tight_layout()
plt.savefig('smart_contract_memory.png', dpi=300, bbox_inches='tight')
print("✓ Memory graph saved: smart_contract_memory.png")
plt.close()

# ========================================================================
# VISUALIZATION 4: COMBINED COMPARISON
# ========================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Comparative Smart Contract Performance Metrics', 
             fontsize=16, fontweight='bold')

colors = ['steelblue', 'darkorange', 'darkgreen']

# Throughput comparison
for idx, (name, data) in enumerate(results.items()):
    axes[0].plot(operation_counts, data['throughput'], 
                marker='o', linewidth=2, label=name.split('(')[0].strip(),
                color=colors[idx])
axes[0].set_ylabel('Throughput (ops/sec)', fontsize=11, fontweight='bold')
axes[0].set_xscale('log')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Latency comparison
for idx, (name, data) in enumerate(results.items()):
    axes[1].plot(operation_counts, data['latency'], 
                marker='s', linewidth=2, label=name.split('(')[0].strip(),
                color=colors[idx])
axes[1].set_ylabel('Latency (ms/operation)', fontsize=11, fontweight='bold')
axes[1].set_xscale('log')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Memory comparison
for idx, (name, data) in enumerate(results.items()):
    axes[2].plot(operation_counts, data['memory'], 
                marker='^', linewidth=2, label=name.split('(')[0].strip(),
                color=colors[idx])
axes[2].set_xlabel('Number of Operations', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Memory Usage (MB)', fontsize=11, fontweight='bold')
axes[2].set_xscale('log')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('smart_contract_combined_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Combined comparison saved: smart_contract_combined_comparison.png")
plt.close()

# ========================================================================
# GENERATE PERFORMANCE SUMMARY TABLE
# ========================================================================

summary_data = []
for name, data in results.items():
    summary_data.append({
        'Contract': name.split('(')[0].strip(),
        'Max Throughput (ops/sec)': f"{max(data['throughput']):.2f}",
        'Min Latency (ms)': f"{min(data['latency']):.4f}",
        'Max Memory (MB)': f"{max(data['memory']):.2f}"
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('smart_contract_performance_summary.csv', index=False)

print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)
print(summary_df.to_string(index=False))
print("="*70)
print("\n✓ Summary table saved: smart_contract_performance_summary.csv")
print("\nAll visualizations generated successfully!")
print("="*70)
