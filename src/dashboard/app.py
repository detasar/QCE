import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.sims.chsh import simulate_chsh
from src.witness.cew import threshold_L, cardinality_product
from src.cp.split import split_conformal_binary
from src.io.config import load_yaml

st.set_page_config(page_title="Quantum Conformal Entanglement", layout="wide")

st.title("Quantum Conformal Entanglement Dashboard")

# Sidebar
st.sidebar.header("Simulation Parameters")
N = st.sidebar.slider("Shots (N)", 100, 10000, 2000)
p_depol = st.sidebar.slider("Depolarizing Noise (p)", 0.0, 0.5, 0.05)
eta = st.sidebar.slider("Detection Efficiency (eta)", 0.5, 1.0, 0.9)
alpha = st.sidebar.slider("Conformal Alpha", 0.01, 0.5, 0.2)

st.sidebar.header("Advanced Noise")
px = st.sidebar.slider("Pauli X Noise", 0.0, 0.2, 0.0)
pz = st.sidebar.slider("Pauli Z Noise", 0.0, 0.2, 0.0)

# Run Simulation
if st.sidebar.button("Run Simulation"):
    st.header("Single Point Simulation Results")
    
    # Config
    cfg = load_yaml('configs/grid_chsh.yml')
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    
    with st.spinner("Simulating..."):
        sim = simulate_chsh(N, angles, 
                            {'p_depol': p_depol, 'px': px, 'pz': pz}, 
                            {'etaA': eta, 'etaB': eta}, 
                            seed=None)
        
        cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], 
                                    sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                    method='knn', K=11, h=0.25, calib='isotonic')
        
        avgX = cp['avg_card_by_ctx'].get(0, 2.0)
        avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
        prod = avgX * avgZ
        
        # CEW Threshold
        cew_cfg = load_yaml('configs/cew.yml')
        L = threshold_L(alpha, cew_cfg['c'], cew_cfg)
        
        is_entangled = prod < L
        
        col1, col2, col3 = st.columns(3)
        col1.metric("CHSH S Value", f"{sim['s_value']:.4f}")
        col2.metric("Cardinality Product", f"{prod:.4f}", delta=f"{L-prod:.4f} margin")
        col3.metric("CEW Result", "Entangled" if is_entangled else "Separable", 
                    delta_color="normal" if is_entangled else "off")
        
        st.subheader("Conformal Sets Analysis")
        st.write(f"Threshold L = {L:.4f}")
        st.write(f"Avg Set Size X: {avgX:.4f}, Z: {avgZ:.4f}")
        
        # Plot p-values
        p0 = cp['pvals_by_ctx'].get(0, [])
        p1 = cp['pvals_by_ctx'].get(1, [])
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(p0, bins=20, ax=ax[0], color='blue')
        ax[0].set_title("P-values (Context 0)")
        sns.histplot(p1, bins=20, ax=ax[1], color='red')
        ax[1].set_title("P-values (Context 1)")
        st.pyplot(fig)

# Load Pre-computed Data
st.header("Phase Diagrams (Pre-computed)")
try:
    df = pd.read_csv('cew_final_map.csv')
    st.write("Loaded `cew_final_map.csv`")
    
    # Heatmap
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    pivot = df.pivot(index='y', columns='x', values='CEW_rate')
    sns.heatmap(pivot, cmap='viridis', ax=ax2)
    ax2.set_title("CEW Rate Phase Diagram")
    ax2.set_xlabel("Depolarizing Noise (p)")
    ax2.set_ylabel("Efficiency (eta)")
    ax2.invert_yaxis()
    st.pyplot(fig2)
    
except FileNotFoundError:
    st.warning("Pre-computed data `cew_final_map.csv` not found. Run experiments to generate it.")

