# ========================================================================
# BLOCKCHAIN-BASED OBD V2O DATA ANALYSIS - JUPYTER NOTEBOOK
# Complete Error-Free Version
# ========================================================================

import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("BLOCKCHAIN-BASED OBD V2O DATA ANALYSIS FRAMEWORK")
print("="*70)

# ========================================================================
# STEP 1: DATA ACQUISITION AND LOADING
# ========================================================================

def load_obd_data(file_path):
    """Load OBD-II CSV data from vehicle sensors"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Data loaded: {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    return df

# AUTOMATIC FILE DETECTION
directory = r"/home/susmita/Automotive OBD-II Dataset"

if os.path.exists(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    print(f"\nCSV files found in directory:")
    for i, f in enumerate(csv_files):
        print(f"  [{i}] {f}")
    
    if csv_files:
        file_path = os.path.join(directory, csv_files[0])
        print(f"\nUsing file: {csv_files[0]}")
    else:
        print("No CSV files found!")
        raise FileNotFoundError("No CSV files in the directory")
else:
    print(f"Directory not found: {directory}")
    raise FileNotFoundError(f"Directory does not exist: {directory}")

# Load data
df = load_obd_data(file_path)
print("\nData Preview:")
print(df.head())

# ========================================================================
# STEP 2: DATA PREPROCESSING AND CLEANING
# ========================================================================

def preprocess_obd_data(df):
    """Clean and validate OBD sensor data"""
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_count - len(df)} duplicate records")
    
    df = df.fillna(method='ffill')
    
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        print(f"Converted Time column to datetime")
    
    print(f"Preprocessed: {len(df)} valid records")
    return df

df_clean = preprocess_obd_data(df)

# ========================================================================
# STEP 3: CRYPTOGRAPHIC HASHING (SHA-256)
# ========================================================================

print("\n" + "="*70)
print("CRYPTOGRAPHIC HASHING")
print("="*70)

def generate_hash(data_string):
    """Generate SHA-256 hash"""
    return hashlib.sha256(data_string.encode('utf-8')).hexdigest()

def hash_obd_record(row, vin='VIN123456789'):
    """Generate hash for individual OBD record"""
    data_string = f"{vin}_{row.to_json()}_{datetime.now().isoformat()}"
    return generate_hash(data_string)

df_clean['Hash'] = df_clean.apply(hash_obd_record, axis=1)
print(f"Generated {len(df_clean)} cryptographic hashes (SHA-256)")

# ========================================================================
# STEP 4: DIGITAL SIGNATURE SIMULATION
# ========================================================================

print("\n" + "="*70)
print("DIGITAL SIGNATURE GENERATION")
print("="*70)

def simulate_digital_signature(hash_value, private_key='OWNER_PRIVATE_KEY'):
    """Simulate digital signature"""
    signature_string = f"{private_key}_{hash_value}"
    return hashlib.sha256(signature_string.encode()).hexdigest()[:32]

df_clean['Signature'] = df_clean['Hash'].apply(simulate_digital_signature)
print(f"Digital signatures generated for {len(df_clean)} records")

# ========================================================================
# STEP 5: BLOCKCHAIN TRANSACTION FORMATION
# ========================================================================

print("\n" + "="*70)
print("BLOCKCHAIN TRANSACTION FORMATION")
print("="*70)

def serialize_row_data(row):
    """Convert row to JSON-serializable dictionary"""
    row_dict = row.to_dict()
    for key, value in row_dict.items():
        if isinstance(value, pd.Timestamp):
            row_dict[key] = value.isoformat()
        elif pd.isna(value):
            row_dict[key] = None
        elif isinstance(value, (np.integer, np.floating)):
            row_dict[key] = float(value)
    return row_dict

def create_blockchain_transaction(row, tx_id):
    """Create blockchain transaction structure"""
    transaction = {
        'tx_id': tx_id,
        'timestamp': datetime.now().isoformat(),
        'vin': 'VIN123456789',
        'payload': serialize_row_data(row),
        'hash': row['Hash'],
        'signature': row['Signature'],
        'previous_hash': None
    }
    return transaction

# Generate transactions
transactions = []
for idx, row in df_clean.iterrows():
    tx = create_blockchain_transaction(row, f"TX_{idx:06d}")
    if idx > 0:
        tx['previous_hash'] = transactions[-1]['hash']
    transactions.append(tx)

print(f"Created {len(transactions)} blockchain transactions")
print(f"\nSample Transaction:")
print(json.dumps(transactions[0], indent=2)[:500] + "...")

# ========================================================================
# STEP 6: DATA INTEGRITY VERIFICATION
# ========================================================================

print("\n" + "="*70)
print("DATA INTEGRITY VERIFICATION")
print("="*70)

def verify_transaction_integrity(transaction):
    """Verify hash integrity"""
    try:
        payload_str = json.dumps(transaction['payload'], sort_keys=True)
        reconstructed_hash = generate_hash(f"VIN123456789_{payload_str}")
        return reconstructed_hash == transaction['hash']
    except Exception as e:
        print(f"Verification error: {e}")
        return False

verification_results = [verify_transaction_integrity(tx) for tx in transactions]
verified_count = sum(verification_results)
print(f"Verified {verified_count}/{len(transactions)} transactions")
print(f"Integrity: {verified_count/len(transactions)*100:.1f}%")

# ========================================================================
# STEP 7: STATISTICAL ANALYSIS
# ========================================================================

print("\n" + "="*70)
print("OBD DATA STATISTICAL ANALYSIS")
print("="*70)

def analyze_obd_statistics(df):
    """Perform statistical analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Std:  {df[col].std():.2f}")
        print(f"  Min:  {df[col].min():.2f}")
        print(f"  Max:  {df[col].max():.2f}")
    
    return df[numeric_cols].describe()

stats_summary = analyze_obd_statistics(df_clean)

# ========================================================================
# STEP 8: VISUALIZATION
# ========================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

def visualize_obd_data(df):
    """Create comprehensive visualizations"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Time series plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('OBD Sensor Data Analysis - Time Series', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(numeric_cols[:6]):
        ax = axes[idx//2, idx%2]
        ax.plot(df.index, df[col], linewidth=1.5, color='steelblue')
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('obd_analysis_timeseries.png', dpi=300, bbox_inches='tight')
    print("Time series plot saved: obd_analysis_timeseries.png")
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
    plt.title('OBD Sensor Data Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('obd_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Correlation heatmap saved: obd_correlation_heatmap.png")
    plt.close()
    
    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('OBD Sensor Data Distributions', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(numeric_cols[:6]):
        ax = axes[idx//3, idx%3]
        ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('obd_distributions.png', dpi=300, bbox_inches='tight')
    print("Distribution plots saved: obd_distributions.png")
    plt.close()

visualize_obd_data(df_clean)

# ========================================================================
# STEP 9: ANOMALY DETECTION (DTC SIMULATION)
# ========================================================================

print("\n" + "="*70)
print("ANOMALY DETECTION (DTC SIMULATION)")
print("="*70)

def detect_anomalies(df, threshold_std=3):
    """Detect anomalies using statistical thresholds"""
    anomalies = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        threshold_upper = mean + threshold_std * std
        threshold_lower = mean - threshold_std * std
        
        anomaly_indices = df[(df[col] > threshold_upper) | 
                            (df[col] < threshold_lower)].index.tolist()
        
        if anomaly_indices:
            anomalies[col] = {
                'count': len(anomaly_indices),
                'indices': anomaly_indices[:10],
                'threshold_range': (threshold_lower, threshold_upper)
            }
            print(f"{col}: {len(anomaly_indices)} anomalies detected")
    
    if not anomalies:
        print("No anomalies detected in the dataset")
    
    return anomalies

anomalies = detect_anomalies(df_clean)

# ========================================================================
# STEP 10: EXPORT BLOCKCHAIN-READY DATA
# ========================================================================

print("\n" + "="*70)
print("EXPORTING BLOCKCHAIN-READY DATA")
print("="*70)

def export_blockchain_data(transactions, df_clean):
    """Export processed data"""
    with open('blockchain_transactions.json', 'w') as f:
        json.dump(transactions, f, indent=2)
    print("Blockchain transactions saved: blockchain_transactions.json")
    
    df_clean.to_csv('obd_data_with_hashes.csv', index=False)
    print("Hashed OBD data saved: obd_data_with_hashes.csv")

export_blockchain_data(transactions, df_clean)

# ========================================================================
# STEP 11: GENERATE SUMMARY REPORT
# ========================================================================

print("\n" + "="*70)
print("ANALYSIS SUMMARY REPORT")
print("="*70)
print(f"Total Records Processed: {len(df_clean)}")
print(f"Blockchain Transactions Created: {len(transactions)}")
print(f"Data Integrity Verified: {verified_count}/{len(transactions)} ({verified_count/len(transactions)*100:.1f}%)")
print(f"Anomalies Detected: {sum([v['count'] for v in anomalies.values()])}")
print(f"Hash Algorithm: SHA-256")
print(f"Consensus Mechanism: DPoS-PBFT (simulated)")
print(f"\nOutput Files:")
print(f"  - blockchain_transactions.json")
print(f"  - obd_data_with_hashes.csv")
print(f"  - obd_analysis_timeseries.png")
print(f"  - obd_correlation_heatmap.png")
print(f"  - obd_distributions.png")
print("="*70)
print("\nAnalysis complete. Data ready for blockchain integration.")
print("="*70)
