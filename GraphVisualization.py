# ========================================================================
# COMPREHENSIVE VISUALIZATION CODE FOR BLOCKCHAIN-BASED OBD ANALYSIS
# This generates publication-quality graphs for each analysis component
# ========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'

# Assuming df_clean, transactions, and anomalies are already created from previous code

# ========================================================================
# GRAPH 1: TIME SERIES ANALYSIS - ALL SENSORS
# ========================================================================

def plot_time_series_analysis(df):
    """Comprehensive time series visualization"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    
    fig, axes = plt.subplots(n_cols, 1, figsize=(15, 3*n_cols))
    fig.suptitle('Complete OBD Sensor Time Series Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx] if n_cols > 1 else axes
        ax.plot(df.index, df[col], linewidth=1.2, color='steelblue', alpha=0.8)
        ax.fill_between(df.index, df[col], alpha=0.3, color='lightblue')
        ax.set_title(f'{col}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics annotation
        mean_val = df[col].mean()
        ax.axhline(mean_val, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('graph_1_timeseries_complete.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 1 saved: graph_1_timeseries_complete.png")
    plt.close()

# ========================================================================
# GRAPH 2: CORRELATION MATRIX HEATMAP
# ========================================================================

def plot_correlation_heatmap(df):
    """Detailed correlation analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    plt.figure(figsize=(14, 12))
    correlation_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlBu_r', center=0, square=True, linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
    
    plt.title('OBD Sensor Correlation Matrix (Lower Triangle)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('graph_2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 2 saved: graph_2_correlation_heatmap.png")
    plt.close()

# ========================================================================
# GRAPH 3: DISTRIBUTION HISTOGRAMS WITH KDE
# ========================================================================

def plot_distributions_with_kde(df):
    """Distribution analysis with kernel density estimation"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    fig.suptitle('OBD Sensor Data Distributions with KDE', 
                 fontsize=18, fontweight='bold')
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Histogram
        ax.hist(df[col].dropna(), bins=40, edgecolor='black', 
                alpha=0.6, color='steelblue', density=True, label='Histogram')
        
        # KDE overlay
        df[col].dropna().plot(kind='kde', ax=ax, color='red', 
                             linewidth=2, label='KDE')
        
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('graph_3_distributions_kde.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 3 saved: graph_3_distributions_kde.png")
    plt.close()

# ========================================================================
# GRAPH 4: BOX PLOTS FOR OUTLIER DETECTION
# ========================================================================

def plot_boxplots(df):
    """Box plot analysis for outlier detection"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('OBD Sensor Outlier Detection - Box Plots', 
                 fontsize=16, fontweight='bold')
    
    # Normalized box plots
    df_normalized = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    df_normalized.boxplot(ax=axes[0], grid=True, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))
    axes[0].set_title('Normalized Data (Z-Score)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Z-Score', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Original scale box plots
    df[numeric_cols].boxplot(ax=axes[1], grid=True, patch_artist=True,
                            boxprops=dict(facecolor='lightgreen', alpha=0.7),
                            medianprops=dict(color='darkred', linewidth=2))
    axes[1].set_title('Original Scale Data', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Value', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('graph_4_boxplots_outliers.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 4 saved: graph_4_boxplots_outliers.png")
    plt.close()

# ========================================================================
# GRAPH 5: PAIRPLOT (SCATTER MATRIX) - First 6 sensors
# ========================================================================

def plot_pairplot(df):
    """Pairwise relationship analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
    
    pairplot_fig = sns.pairplot(df[numeric_cols], diag_kind='kde', 
                                plot_kws={'alpha': 0.6, 's': 20, 'color': 'steelblue'},
                                diag_kws={'color': 'red', 'linewidth': 2})
    pairplot_fig.fig.suptitle('OBD Sensor Pairwise Relationships (Top 6 Sensors)', 
                             fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig('graph_5_pairplot_relationships.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 5 saved: graph_5_pairplot_relationships.png")
    plt.close()

# ========================================================================
# GRAPH 6: ANOMALY DETECTION VISUALIZATION
# ========================================================================

def plot_anomaly_detection(df, anomalies):
    """Visualize detected anomalies"""
    if not anomalies:
        print("No anomalies to visualize")
        return
    
    anomaly_cols = list(anomalies.keys())[:6]  # First 6 columns with anomalies
    n_cols = len(anomaly_cols)
    
    if n_cols == 0:
        print("No anomalies detected in the dataset")
        return
    
    fig, axes = plt.subplots(n_cols, 1, figsize=(15, 4*n_cols))
    fig.suptitle('Anomaly Detection Results (3-Sigma Rule)', 
                 fontsize=16, fontweight='bold')
    
    axes = [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(anomaly_cols):
        ax = axes[idx]
        
        # Plot normal data
        ax.plot(df.index, df[col], linewidth=1, color='steelblue', 
                alpha=0.7, label='Normal Data')
        
        # Highlight anomalies
        anomaly_indices = anomalies[col]['indices']
        ax.scatter(anomaly_indices, df.loc[anomaly_indices, col], 
                  color='red', s=50, marker='X', label='Anomalies', zorder=5)
        
        # Threshold lines
        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.axhline(mean_val + 3*std_val, color='orange', linestyle='--', 
                  alpha=0.7, label='Upper Threshold (+3σ)')
        ax.axhline(mean_val - 3*std_val, color='orange', linestyle='--', 
                  alpha=0.7, label='Lower Threshold (-3σ)')
        ax.axhline(mean_val, color='green', linestyle='-', 
                  alpha=0.5, label='Mean')
        
        ax.set_title(f'{col} - {anomalies[col]["count"]} anomalies detected', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graph_6_anomaly_detection.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 6 saved: graph_6_anomaly_detection.png")
    plt.close()

# ========================================================================
# GRAPH 7: BLOCKCHAIN TRANSACTION CHAIN VISUALIZATION
# ========================================================================

def plot_blockchain_chain(transactions):
    """Visualize blockchain transaction chain"""
    sample_size = min(20, len(transactions))
    sample_tx = transactions[:sample_size]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for idx, tx in enumerate(sample_tx):
        # Block rectangle
        rect = Rectangle((idx*0.8, 0), 0.7, 1, 
                        facecolor='lightblue', edgecolor='darkblue', 
                        linewidth=2)
        ax.add_patch(rect)
        
        # Block text
        ax.text(idx*0.8 + 0.35, 0.7, f"Block {idx}", 
               ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(idx*0.8 + 0.35, 0.5, tx['tx_id'][:8], 
               ha='center', va='center', fontsize=7)
        ax.text(idx*0.8 + 0.35, 0.3, f"Hash: {tx['hash'][:6]}...", 
               ha='center', va='center', fontsize=6)
        
        # Chain arrow
        if idx < sample_size - 1:
            ax.arrow(idx*0.8 + 0.7, 0.5, 0.05, 0, 
                    head_width=0.15, head_length=0.03, 
                    fc='darkgreen', ec='darkgreen')
    
    ax.set_xlim(-0.2, sample_size*0.8)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')
    ax.set_title(f'Blockchain Transaction Chain (First {sample_size} Blocks)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('graph_7_blockchain_chain.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 7 saved: graph_7_blockchain_chain.png")
    plt.close()

# ========================================================================
# GRAPH 8: HASH DISTRIBUTION ANALYSIS
# ========================================================================

def plot_hash_distribution(df):
    """Analyze hash value distribution"""
    # Convert first 8 characters of hash to hex values
    hash_values = df['Hash'].apply(lambda x: int(x[:8], 16))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cryptographic Hash Distribution Analysis (SHA-256)', 
                 fontsize=14, fontweight='bold')
    
    # Histogram
    axes[0, 0].hist(hash_values, bins=50, edgecolor='black', 
                   alpha=0.7, color='purple')
    axes[0, 0].set_title('Hash Value Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Hash Value (First 8 Hex Digits)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # KDE
    hash_values.plot(kind='kde', ax=axes[0, 1], color='darkred', linewidth=2)
    axes[0, 1].set_title('Hash Value Density', fontweight='bold')
    axes[0, 1].set_xlabel('Hash Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hexbin
    axes[1, 0].hexbin(range(len(hash_values)), hash_values, 
                     gridsize=30, cmap='YlOrRd')
    axes[1, 0].set_title('Hash Value Hexbin Plot', fontweight='bold')
    axes[1, 0].set_xlabel('Transaction Index')
    axes[1, 0].set_ylabel('Hash Value')
    
    # Time series
    axes[1, 1].plot(hash_values.values, linewidth=1, color='teal')
    axes[1, 1].set_title('Hash Value Sequence', fontweight='bold')
    axes[1, 1].set_xlabel('Transaction Index')
    axes[1, 1].set_ylabel('Hash Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graph_8_hash_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 8 saved: graph_8_hash_distribution.png")
    plt.close()

# ========================================================================
# GRAPH 9: STATISTICAL SUMMARY VISUALIZATION
# ========================================================================

def plot_statistical_summary(df):
    """Visual statistical summary"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe().T
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OBD Data Statistical Summary', fontsize=16, fontweight='bold')
    
    # Mean comparison
    stats['mean'].plot(kind='barh', ax=axes[0, 0], color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Mean Values', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Mean')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Standard deviation
    stats['std'].plot(kind='barh', ax=axes[0, 1], color='orange', alpha=0.7)
    axes[0, 1].set_title('Standard Deviation', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Std Dev')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Min-Max range
    stats[['min', 'max']].plot(kind='barh', ax=axes[1, 0], alpha=0.7)
    axes[1, 0].set_title('Min-Max Range', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].legend(['Min', 'Max'])
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Quartiles
    stats[['25%', '50%', '75%']].plot(kind='barh', ax=axes[1, 1], alpha=0.7)
    axes[1, 1].set_title('Quartile Distribution', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].legend(['Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)'])
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('graph_9_statistical_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 9 saved: graph_9_statistical_summary.png")
    plt.close()

# ========================================================================
# GRAPH 10: DATA INTEGRITY VERIFICATION RESULTS
# ========================================================================

def plot_integrity_verification(verification_results):
    """Visualize blockchain integrity verification"""
    verified = sum(verification_results)
    total = len(verification_results)
    failed = total - verified
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Blockchain Data Integrity Verification Results', 
                 fontsize=14, fontweight='bold')
    
    # Pie chart
    colors = ['#2ecc71', '#e74c3c']
    labels = [f'Verified\n({verified})', f'Failed\n({failed})']
    axes[0].pie([verified, failed], labels=labels, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('Verification Status Distribution', fontweight='bold')
    
    # Bar chart
    axes[1].bar(['Verified', 'Failed'], [verified, failed], 
               color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Number of Transactions', fontsize=11)
    axes[1].set_title('Transaction Verification Count', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate([verified, failed]):
        axes[1].text(i, v + max(verified, failed)*0.02, str(v), 
                    ha='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('graph_10_integrity_verification.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 10 saved: graph_10_integrity_verification.png")
    plt.close()

# ========================================================================
# EXECUTE ALL VISUALIZATIONS
# ========================================================================

print("\n" + "="*70)
print("GENERATING ALL PUBLICATION-QUALITY GRAPHS")
print("="*70 + "\n")

# Generate all graphs
plot_time_series_analysis(df_clean)
plot_correlation_heatmap(df_clean)
plot_distributions_with_kde(df_clean)
plot_boxplots(df_clean)
plot_pairplot(df_clean)
plot_anomaly_detection(df_clean, anomalies)
plot_blockchain_chain(transactions)
plot_hash_distribution(df_clean)
plot_statistical_summary(df_clean)
plot_integrity_verification(verification_results)

print("\n" + "="*70)
print("ALL GRAPHS GENERATED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files:")
print("  1. graph_1_timeseries_complete.png")
print("  2. graph_2_correlation_heatmap.png")
print("  3. graph_3_distributions_kde.png")
print("  4. graph_4_boxplots_outliers.png")
print("  5. graph_5_pairplot_relationships.png")
print("  6. graph_6_anomaly_detection.png")
print("  7. graph_7_blockchain_chain.png")
print("  8. graph_8_hash_distribution.png")
print("  9. graph_9_statistical_summary.png")
print(" 10. graph_10_integrity_verification.png")
print("="*70)
