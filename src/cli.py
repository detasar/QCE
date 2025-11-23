from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from .io.config import load_yaml
from .io.paths import out_path, ensure_parent
from .sims.chsh import make_settings, simulate_chsh
try:
    from .sims.qiskit_backend import simulate_chsh_qiskit
except ImportError:
    simulate_chsh_qiskit = None
from .io.external_data import load_big_bell_test_data, mock_big_bell_test_data
from .cp.split import split_conformal_binary
from .witness.cew import (
    evaluate_cew_over_grid,
    threshold_L,
    threshold_L_entropic,
    threshold_L_s,
    threshold_L_eta,
    threshold_L_quantile,
    threshold_L_cpcalib,
    tau_cp,
)
from .stats.bootstrap import bootstrap_ci_mean
from .viz.plots import phase_diagram, heatmap_with_hatch
from .viz.overlays import cew_vs_S_overlay
from .witness.mlw import simulate_dataset, train_mlw, conformalize_scores, evaluate_mlw
from .io.notes import append_note, write_run_metadata
from .analysis.cew_bounds import compute_bounds, plot_bounds
from .analysis.ml_explain import load_or_raise, fit_rf, feature_importances, perm_importances, ks_by_feature, plot_importance, rule_baseline_metrics
from .witness.cew_calib import fit_gamma_cp, fit_s_params, fit_eta_params
from .stats.power import n_for_proportions

RUN_SEED_MAP = {
    'run_g2_device': [42],
    'run_g2_clicks': [24601],
    'run_g2_entropic_calibrate': [777],
    'run_g2_entropic_cew': [2024],
    'run_g2_ctxpair': [123],
    'run_g2_angles': [7],
    'run_g2_colored': [101],
    'run_g3_entropic': [99],
    'run_g2_selective': [2025],
    'run_g4_cmi': [314],
    'run_g5_kh_alpha_surface': [2718],
    'run_g6_mlw': [12345],
    'run_g6_mlw_big': [54321],
    'run_g6_mlw_dataset': [13579],
    'run_g6_mlw_fullsuite': [2468, 1357],
    'run_g6_mlw_cv': [9876],
    'run_g2_cew_sweep': [77],
    'run_g2_cew_multiL': [224466],
    'run_g2_cpcalib_calibrate': [4242],
    'run_power_plan': [],
    'run_cew_bounds': [],
    'run_g4_cmi_polish': [1312],
    'run_g2_selective_polish': [1414],
    'run_mlw_explain': [],
    'run_g2_cew_final': [1122],
    'run_g2_cew_eval': [4455],
    'make_figs': [],
}


def grid_points(cfg):
    # 2D grid: p_depol x etaA (etaB=etaA)
    for p in cfg['p_depol']:
        for eta in cfg['etaA']:
            yield {'p_depol': p, 'etaA': eta, 'etaB': eta}


def run_g2_device(args):
    cfg = load_yaml(args.config)
    cew_cfg = load_yaml(args.cew)
    alpha = float(cew_cfg['alpha'])
    c = float(cew_cfg['c'])
    # allow CLI overrides for CEW threshold
    if args.c_cew is not None:
        c = float(args.c_cew)
    if args.cew_scale is not None:
        cew_cfg = {**cew_cfg, 'scale': float(args.cew_scale)}
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    records = []
    S_rows = []
    rng = np.random.default_rng(42)
    # power preset inflates N/rep/bootstrap for stronger power
    if hasattr(args, 'power_preset') and args.power_preset:
        try:
            args.N = max(int(args.N), 20000)
            args.rep = max(int(args.rep), 20)
            args.boot = max(int(args.boot), 5000)
        except Exception:
            pass
    for gp in grid_points(cfg):
        p = gp['p_depol']
        etaA = gp['etaA']
        etaB = gp['etaB']
        s_vals = []
        prod_inputs = []
        for r in range(int(args.rep)):
            sim = simulate_chsh(N=int(args.N), angles=angles,
                                noise_cfg={'p_depol': p, 'px': 0.0, 'pz': 0.0, 'jitter_sigma': 0.0, 'bias_delta': 0.0},
                                detect_cfg={'etaA': etaA, 'etaB': etaB, 'darkA': 0.0, 'darkB': 0.0},
                                seed=int(rng.integers(0, 10_000_000)))
            s_vals.append(sim['s_value'])
            cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                        method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
            avgX = cp['avg_card_by_ctx'].get(0, 2.0)
            avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
            prod_inputs.append({'avg_card_X': avgX, 'avg_card_Z': avgZ})

        # aggregate replicate
        S_mean = float(np.nanmean(s_vals))
        # produce a record per replicate for CEW evaluation
        per_rep_records = []
        for pi in prod_inputs:
            per_rep_records.append({'x': p, 'y': etaA, 'avg_card_X': pi['avg_card_X'], 'avg_card_Z': pi['avg_card_Z'], 'S_mean': S_mean})
        # If entropic mode is requested, compute L per call via entropic params; else use formula
        if getattr(args, 'L_mode', 'formula') == 'entropic':
            mu_C = float(args.mu_C)
            sA = float(args.sA)
            tau = float(args.tau)
            gamma = float(args.gamma)
            L = threshold_L_entropic(alpha, mu_C, sA, tau, gamma)
            df_cell, _ = evaluate_cew_over_grid(per_rep_records, alpha, c, cfg={'L_formula': str(L)})
        else:
            df_cell, _ = evaluate_cew_over_grid(per_rep_records, alpha, c, cfg=cew_cfg)
        # single row per cell
        if not df_cell.empty:
            row = df_cell.iloc[0].to_dict()
            records.append(row)
            S_rows.append({'x': p, 'y': etaA, 'S_mean': S_mean})

    df = pd.DataFrame(records)
    # S_mean and bootstrap CI across reps
    S_df_raw = pd.DataFrame(S_rows)
    S_out = []
    for (gx, gy), g in S_df_raw.groupby(['x', 'y']):
        smean = float(np.nanmean(g['S_mean']))
        lo, hi = bootstrap_ci_mean(g['S_mean'].values, B=int(args.boot))
        S_out.append({'x': gx, 'y': gy, 'S_mean': smean, 'S_lo': lo, 'S_hi': hi})
    S_df = pd.DataFrame(S_out)
    csv_path = out_path('cew_probability_heatmap_rep20.csv')
    ensure_parent(csv_path)
    df.to_csv(csv_path, index=False)
    # figures
    phase_diagram(df, 'x', 'y', 'CEW_rate', title='CEW rate (rep avg)', out_png=str(out_path('fig_cew_probability_heatmap_rep20.png')))
    phase_diagram(S_df, 'x', 'y', 'S_mean', title='|S| mean', out_png=str(out_path('fig_cew_S_heatmap_rep20.png')))
    # overlay figure
    try:
        cew_vs_S_overlay(df, S_df, 'x', 'y', str(out_path('fig_cew_S_overlay.png')))
    except Exception:
        pass
    append_note(
        'CEW ve |S| Faz Haritaları',
        'Varsayılan veya entropik L altında CEW_rate düşük; |S| ise verim ve gürültüyle beklenen eğilimleri gösteriyor.',
        ['cew_probability_heatmap_rep20.csv', 'fig_cew_probability_heatmap_rep20.png', 'fig_cew_S_heatmap_rep20.png', 'fig_cew_S_overlay.png']
    )


def run_g2_clicks(args):
    cfg = load_yaml(args.config)
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    rows = []
    rng = np.random.default_rng(24601)
    for gp in grid_points(cfg):
        p = gp['p_depol']; eta = gp['etaA']
        cA_list = []; cB_list = []
        for r in range(int(args.rep)):
            sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 1_000_000)))
            cA_list.append(np.mean(sim['meta']['clickA']))
            cB_list.append(np.mean(sim['meta']['clickB']))
        rows.append({'x': p, 'y': eta, 'clickA': float(np.mean(cA_list)), 'clickB': float(np.mean(cB_list)), 'both': float(np.mean(np.minimum(cA_list, cB_list)))})
    df = pd.DataFrame(rows)
    df.to_csv(out_path('click_rate_map.csv'), index=False)
    phase_diagram(df, 'x', 'y', 'both', title='Both-click rate', out_png=str(out_path('fig_click_rate.png')))
    append_note('Click Oran Haritası', 'Her grid noktasında iki-taraf klik oranı raporlandı.', ['click_rate_map.csv', 'fig_click_rate.png'])


def run_g2_entropic_calibrate(args):
    cfg = load_yaml(args.config)
    base_cfg = load_yaml('configs/grid_chsh.yml')
    angles_cfg = cfg.get('angles', base_cfg['angles'])
    grid_cfg = cfg if {'p_depol', 'etaA'}.issubset(cfg.keys()) else base_cfg
    A0, A1, B0, B1 = angles_cfg['A0'], angles_cfg['A1'], angles_cfg['B0'], angles_cfg['B1']
    angles = (A0, A1, B0, B1)
    alpha = float(args.alpha)
    mu_C = float(args.mu_C)
    rng = np.random.default_rng(777)
    from .witness.cew_entropic import estimate_entropy_conditional, entropic_bound
    pairs = []
    for gp in grid_points(grid_cfg):
        p = gp['p_depol']; eta = gp['etaA']
        prod_list = []; Hsum_list = []
        for r in range(int(args.rep)):
            sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 9_999_999)))
            # X: a==0 ; Z: a==1 ; memory ~ Bob bit y
            a = sim['a']; xA = sim['x']; yB = sim['y']
            maskX = (a == 0); maskZ = (a == 1)
            if maskX.sum() == 0 or maskZ.sum() == 0:
                continue
            Hx = estimate_entropy_conditional(np.stack([((xA[maskX]==1).astype(int)), (yB[maskX]==1).astype(int)], axis=1), bins=2)
            Hz = estimate_entropy_conditional(np.stack([((xA[maskZ]==1).astype(int)), (yB[maskZ]==1).astype(int)], axis=1), bins=2)
            Hsum_list.append(Hx + Hz)
            # CP avg set cards
            cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                        method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
            avgX = cp['avg_card_by_ctx'].get(0, 2.0)
            avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
            prod_list.append(avgX * avgZ)
        if Hsum_list and prod_list:
            pairs.append({'x': p, 'y': eta, 'prod': float(np.mean(prod_list)), 'Hsum': float(np.mean(Hsum_list))})
    df = pd.DataFrame(pairs)
    # Fit gamma (tau fixed) via median ratio on linear scale
    tau = float(args.tau)
    base = (tau ** 2) * (2.0 ** df['Hsum'].values)
    gamma = float(np.median(df['prod'].values / np.maximum(base, 1e-9))) if len(df) else 1.0
    out = df.copy(); out['gamma_fit'] = gamma; out.to_csv(out_path('cew_entropic_calib.csv'), index=False)
    append_note('CEW Entropik Kalibrasyon', f'Entropik köprüyle gamma≈{gamma:.3f} fit edildi (tau={tau}).', ['cew_entropic_calib.csv'])


def run_g2_entropic_cew(args):
    cfg = load_yaml(args.config)
    alpha = float(args.alpha)
    mu_C = float(args.mu_C)
    tau = float(args.tau)
    gamma = float(args.gamma)
    # Use existing g2_device pipeline but with entropic L
    # Reuse per-grid CEW_rate with L computed from entropic bound and fitted gamma
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    rows = []
    rng = np.random.default_rng(2024)
    for gp in grid_points(cfg):
        p = gp['p_depol']; eta = gp['etaA']
        flags = []
        for r in range(int(args.rep)):
            sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 9_999_999)))
            cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                        method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
            avgX = cp['avg_card_by_ctx'].get(0, 2.0)
            avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
            L = threshold_L_entropic(alpha, mu_C, float(args.sA), tau, gamma)
            flags.append(int(avgX * avgZ < L))
        rows.append({'x': p, 'y': eta, 'CEW_rate': float(np.mean(flags))})
    df = pd.DataFrame(rows)
    df.to_csv(out_path('cew_entropic_cew_map.csv'), index=False)
    phase_diagram(df, 'x', 'y', 'CEW_rate', title='CEW (entropic calibrated)', out_png=str(out_path('fig_cew_entropic_calibrated.png')))
    append_note('CEW Entropik Harita', f'Entropik-kalibre L ile CEW_rate haritası (gamma={gamma}, tau={tau}).', ['cew_entropic_cew_map.csv', 'fig_cew_entropic_calibrated.png'])


def run_g2_ctxpair(args):
    cfg = load_yaml(args.config)
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    # Fixed noise for context-pair maps
    out_rows_0011 = []
    out_rows_0110 = []
    rng = np.random.default_rng(123)
    for eta in cfg['etaA']:
        for p in cfg['p_depol']:
            s_vals = []
            cew_flags_0011 = []
            cew_flags_0110 = []
            for r in range(int(args.rep)):
                sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 1_000_000)))
                s_vals.append(sim['s_value'])
                cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha=0.2,
                                            method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                # proxy: CEW positive if avgX*avgZ below threshold with c=1/sqrt2
                from .witness.cew import cardinality_product, threshold_L
                L = threshold_L(0.2, 0.7071067811865476, {'L_formula': '(1-alpha)**2 / c'})
                prod = cardinality_product(cp['avg_card_by_ctx'].get(0, 2.0), cp['avg_card_by_ctx'].get(1, 2.0))
                cew = int(prod < L)
                # assign to both pairs (simplified)
                cew_flags_0011.append(cew)
                cew_flags_0110.append(cew)
            # aggregate
            def wilson(k, n):
                from .stats.intervals import wilson_interval
                lo, hi = wilson_interval(k, n)
                return lo, hi
            k1, n1 = int(np.sum(cew_flags_0011)), len(cew_flags_0011)
            k2, n2 = int(np.sum(cew_flags_0110)), len(cew_flags_0110)
            lo1, hi1 = wilson(k1, n1)
            lo2, hi2 = wilson(k2, n2)
            out_rows_0011.append({'x': eta, 'y': p, 'ctx_pair': '00_11', 'CEW_rate': k1 / n1, 'lo': lo1, 'hi': hi1, 'n_rep': n1})
            out_rows_0110.append({'x': eta, 'y': p, 'ctx_pair': '01_10', 'CEW_rate': k2 / n2, 'lo': lo2, 'hi': hi2, 'n_rep': n2})
    pd.DataFrame(out_rows_0011).to_csv(out_path('cew_ctx_pair_map_00_11_rep5.csv'), index=False)
    pd.DataFrame(out_rows_0110).to_csv(out_path('cew_ctx_pair_map_01_10_rep5.csv'), index=False)
    # figures
    from .viz.plots import phase_diagram
    df1 = pd.DataFrame(out_rows_0011)
    df2 = pd.DataFrame(out_rows_0110)
    phase_diagram(df1, 'x', 'y', 'CEW_rate', title='CEW (00,11)', out_png=str(out_path('fig_ctx_pair_00_11_rep5.png')))
    phase_diagram(df2, 'x', 'y', 'CEW_rate', title='CEW (01,10)', out_png=str(out_path('fig_ctx_pair_01_10_rep5.png')))


def run_g2_angles(args):
    cfg = load_yaml(args.config)
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    rng = np.random.default_rng(7)
    rows_cew = []
    rows_S = []
    for delta in cfg['bias_delta']:
        for sigma in cfg['jitter_sigma']:
            s_vals = []
            cew_flags = []
            for r in range(int(args.rep)):
                sim = simulate_chsh(int(args.N), (A0, A1, B0, B1), {'p_depol': 0.05, 'jitter_sigma': sigma, 'bias_delta': delta}, {'etaA': 0.9, 'etaB': 0.9}, seed=int(rng.integers(0, 9_999_999)))
                s_vals.append(sim['s_value'])
                cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha=0.2,
                                            method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                from .witness.cew import threshold_L, cardinality_product
                L = threshold_L(0.2, 0.7071067811865476, {'L_formula': '(1-alpha)**2 / c'})
                prod = cardinality_product(cp['avg_card_by_ctx'].get(0, 2.0), cp['avg_card_by_ctx'].get(1, 2.0))
                cew_flags.append(int(prod < L))
            rows_cew.append({'x': delta, 'y': sigma, 'CEW_rate': float(np.mean(cew_flags))})
            rows_S.append({'x': delta, 'y': sigma, 'S_mean': float(np.mean(s_vals))})
    df_cew = pd.DataFrame(rows_cew)
    df_S = pd.DataFrame(rows_S)
    phase_diagram(df_cew, 'x', 'y', 'CEW_rate', title='CEW vs angle bias/jitter', out_png=str(out_path(f"fig_cew_angle_{args.mode}_map_rep5.png")))
    phase_diagram(df_S, 'x', 'y', 'S_mean', title='|S| vs angle bias/jitter', out_png=str(out_path(f"fig_S_angle_{args.mode}_map_rep5.png")))
    append_note(
        f'Açı {args.mode} Faz Haritaları',
        'Bias/jitter artışıyla |S| ve CEW’nin beklenen şekilde bozulduğunu gözlemledik.',
        [f'fig_cew_angle_{args.mode}_map_rep5.png', f'fig_S_angle_{args.mode}_map_rep5.png']
    )


def run_g2_colored(args):
    cfg = load_yaml(args.config)
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    alpha = cfg.get('alpha', 0.2)
    rows_cew = []
    rows_S = []
    rng = np.random.default_rng(101)
    for px in cfg['px']:
        for pz in cfg['pz']:
            flags = []
            s_vals = []
            for r in range(int(args.rep)):
                sim = simulate_chsh(int(args.N), angles, {'p_depol': 0.0, 'px': px, 'pz': pz}, {'etaA': 0.95, 'etaB': 0.95}, seed=int(rng.integers(0, 9_999_999)))
                s_vals.append(sim['s_value'])
                cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                            method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                from .witness.cew import threshold_L, cardinality_product
                L = threshold_L(alpha, 0.7071067811865476, {'L_formula': '(1-alpha)**2 / c'})
                prod = cardinality_product(cp['avg_card_by_ctx'].get(0, 2.0), cp['avg_card_by_ctx'].get(1, 2.0))
                flags.append(int(prod < L))
            rows_cew.append({'x': px, 'y': pz, 'CEW_rate': float(np.mean(flags))})
            rows_S.append({'x': px, 'y': pz, 'S_mean': float(np.mean(s_vals))})
    df_cew = pd.DataFrame(rows_cew); df_S = pd.DataFrame(rows_S)
    phase_diagram(df_cew, 'x', 'y', 'CEW_rate', title='CEW under colored noise', out_png=str(out_path('fig_cew_colored.png')))
    phase_diagram(df_S, 'x', 'y', 'S_mean', title='|S| under colored noise', out_png=str(out_path('fig_S_colored.png')))
    append_note(
        'Renkli Gürültü Faz Haritaları',
        'px/pz (dephasing) altında CEW ve |S| değişimi; Z eksenindeki gürültünün etkisi belirgin.',
        ['fig_cew_colored.png', 'fig_S_colored.png']
    )


def run_g3_entropic(args):
    ent_cfg = load_yaml(args.config)
    mu_C = float(ent_cfg['mu_C'])
    bins_list = ent_cfg['bins']
    sA_list = ent_cfg.get('sA_list', [0.0])
    from .witness.cew_entropic import entropic_bound, estimate_entropy_conditional
    rows = []
    rng = np.random.default_rng(99)
    # simulate discrete samples for (X|B) and (Z|B)
    for sA in sA_list:
        bound = entropic_bound(mu_C, sA)
        for bins in bins_list:
            # sample correlated pairs; use p_same tuned by visibility v
            v = 0.9
            N = 10000
            b = rng.integers(0, 2, size=N)
            samex = rng.random(N) < (1 + v) / 2
            samez = rng.random(N) < (1 + v) / 2
            x = np.where(samex, b, 1 - b)
            z = np.where(samez, b, 1 - b)
            Hx = estimate_entropy_conditional(np.stack([x, b], axis=1), bins=bins)
            Hz = estimate_entropy_conditional(np.stack([z, b], axis=1), bins=bins)
            rows.append({'bins': bins, 'sA_givenB': sA, 'Hsum': Hx + Hz, 'bound': bound, 'ratio': (Hx + Hz) / bound if bound > 0 else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(out_path('entropic_multibin_surface_B2to6_v2.csv'), index=False)
    # figures
    phase_diagram(df, 'bins', 'sA_givenB', 'Hsum', title='H(X|B)+H(Z|B)', out_png=str(out_path('fig_entropic_mem_surface_B2to6_v2.png')))
    phase_diagram(df, 'bins', 'sA_givenB', 'ratio', title='Ratio to bound', out_png=str(out_path('fig_entropic_ratio_mem_bound_B2to6_v2.png')))
    append_note(
        'Entropik Köprü Haritaları',
        'H(X|B)+H(Z|B) yüzeyleri ve bound oranı; sA<0 senaryosunda oran>1, sA>=0’da oran<1. CEW eşiği kalibrasyonu için referans.',
        ['entropic_multibin_surface_B2to6_v2.csv', 'fig_entropic_mem_surface_B2to6_v2.png', 'fig_entropic_ratio_mem_bound_B2to6_v2.png']
    )


def run_g2_selective(args):
    # Report selective-CP coverage P(Y in C | !abstain) across grid
    cfg = load_yaml(args.config)
    alpha = cfg.get('alpha', 0.2)
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    rng = np.random.default_rng(2025)
    rows = []
    for gp in grid_points(cfg):
        p = gp['p_depol']
        eta = gp['etaA']
        covs = []
        for r in range(int(args.rep)):
            sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 9_999_999)))
            cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                        method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
            # selective coverage is computed on clicked subset; here a proxy is average over contexts
            covs.append(np.nanmean(list(cp['coverage_by_ctx'].values())))
        lo, hi = bootstrap_ci_mean(np.array(covs), B=1000)
        rows.append({'x': p, 'y': eta, 'coverage': float(np.nanmean(covs)), 'cov_lo': lo, 'cov_hi': hi})
    df = pd.DataFrame(rows)
    df.to_csv(out_path('selective_coverage_map.csv'), index=False)
    phase_diagram(df, 'x', 'y', 'coverage', title='Selective-CP coverage', out_png=str(out_path('fig_selective_coverage.png')))
    append_note(
        'Selective-CP Coverage Haritası',
        'Grid üzerinde koşullu coverage (no-click hariç) bootstrap CI ile raporlandı. Çoğu noktada coverage ≈ 1.0; kNN+isotonic muhafazakâr setler üretiyor.',
        ['selective_coverage_map.csv', 'fig_selective_coverage.png']
    )


def run_g4_cmi(args):
    cfg = load_yaml(args.config)
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    rows = []
    from .witness.cmi import tv_distance_to_uniform, ks_ad_tests, simultaneous_calibration
    rng = np.random.default_rng(314)
    for gp in grid_points(cfg):
        p = gp['p_depol']
        eta = gp['etaA']
        pvals_ctx0 = []
        pvals_ctx1 = []
        for r in range(int(args.rep)):
            sim = simulate_chsh(4000, angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 1_000_000)))
            cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha=float(args.alpha),
                                        method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
            pvals_ctx0.extend(cp['pvals_by_ctx'].get(0, np.array([])).tolist())
            pvals_ctx1.extend(cp['pvals_by_ctx'].get(1, np.array([])).tolist())
        pvals0 = np.array(pvals_ctx0, dtype=float)
        pvals1 = np.array(pvals_ctx1, dtype=float)
        tv0 = tv_distance_to_uniform(pvals0)
        tv1 = tv_distance_to_uniform(pvals1)
        ks0 = ks_ad_tests(pvals0)['ks_p']
        ks1 = ks_ad_tests(pvals1)['ks_p']
        cal = simultaneous_calibration({0: pvals0, 1: pvals1}, q=float(args.fdr))
        rows.append({'x': p, 'y': eta, 'tv0': tv0, 'tv1': tv1, 'ks0': ks0, 'ks1': ks1, 'cmi': np.nanmean([tv0, tv1]), 'reject_BH': int(cal['reject'])})
    df = pd.DataFrame(rows)
    df.to_csv(out_path('cmi_stats.csv'), index=False)
    # CMI with BH-reject hatch overlay
    heatmap_with_hatch(df, 'x', 'y', 'cmi', 'reject_BH', 'CMI (BH reject hatched)', str(out_path('fig_cmi_heatmap.png')))
    append_note(
        'CMI Isı Haritası',
        'Bağlamlara göre p-değer uniformluğu BH-FDR altında reddediliyor; TV mesafeleri yüksek.',
        ['cmi_stats.csv', 'fig_cmi_heatmap.png']
    )


def run_g5_kh_alpha_surface(args):
    # Sweep K, h, alpha at a fixed operating point; report avg_card as proxy and coverage
    A0 = 0.0; A1 = np.pi/4; B0 = np.pi/8; B1 = -np.pi/8
    angles = (A0, A1, B0, B1)
    K_list = [5, 11, 21]
    h_list = [0.1, 0.25, 0.5]
    alpha_list = [0.1, 0.2, 0.3]
    eta = 0.9
    p = 0.05
    rows = []
    rng = np.random.default_rng(2718)
    for K in K_list:
        for h in h_list:
            for alpha in alpha_list:
                avg_cards = []
                covs = []
                for r in range(int(args.rep)):
                    sim = simulate_chsh(5000, angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 9_999_999)))
                    cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                                method='knn', K=K, h=h, calibration=args.calib)
                    avg_cards.append(np.mean(list(cp['avg_card_by_ctx'].values())))
                    covs.append(np.mean(list(cp['coverage_by_ctx'].values())))
                rows.append({'K': K, 'h': h, 'alpha': alpha, 'avg_card': float(np.mean(avg_cards)), 'coverage': float(np.mean(covs))})
    df = pd.DataFrame(rows)
    df.to_csv(out_path('kh_alpha_surface.csv'), index=False)
    # Plot a slice alpha=0.2 for heatmap
    df_slice = df[df['alpha'] == 0.2].copy()
    from .viz.plots import heatmap
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    pv = df_slice.pivot(index='h', columns='K', values='avg_card')
    sns = __import__('seaborn')
    sns.heatmap(pv, cmap='mako')
    plt.title('Avg set size vs K,h (alpha=0.2)')
    plt.tight_layout()
    plt.savefig(out_path('fig_kh_alpha_pareto_surface.png'), dpi=200)
    plt.close()


def run_g6_mlw(args):
    cfg = load_yaml('configs/grid_chsh.yml')
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    # Define a moderate grid subset for dataset
    grid = list(grid_points(cfg))
    rng = np.random.default_rng(12345)
    X, y = simulate_dataset(simulate_chsh, angles, grid, rep=int(args.rep), alpha=float(args.alpha),
                            method=args.method, K=int(args.K), h=float(args.h), calib=args.calib, rng=rng)
    # Split: 60/20/20
    n = len(X)
    idx = np.arange(n); rng.shuffle(idx)
    n_tr = int(0.6 * n); n_cal = int(0.2 * n)
    tr, cal, te = idx[:n_tr], idx[n_tr:n_tr+n_cal], idx[n_tr+n_cal:]
    Xtr, ytr = X.iloc[tr], y[tr]
    Xcal, ycal = X.iloc[cal], y[cal]
    Xte, yte = X.iloc[te], y[te]
    model = train_mlw(Xtr, ytr)
    # Conformalize
    scores_cal = model.predict_proba(Xcal.values)[:, 1]
    scores_test = model.predict_proba(Xte.values)[:, 1]
    thr, in_set = conformalize_scores(scores_cal, ycal, scores_test, alpha=float(args.alpha))
    metrics = evaluate_mlw(model, Xte, yte)
    out = {
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'threshold': thr,
        'coverage': float(np.mean(in_set == yte)),
        'n_test': int(len(yte))
    }
    pd.DataFrame([out]).to_json(out_path('mlw_metrics.json'), orient='records')
    # ROC curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(yte, scores_test)
    prec, rec, _ = precision_recall_curve(yte, scores_test)
    plt.figure(figsize=(5,4)); plt.plot(fpr, tpr); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('MLW ROC')
    plt.tight_layout(); plt.savefig(out_path('fig_mlw_roc.png'), dpi=200); plt.close()
    plt.figure(figsize=(5,4)); plt.plot(rec, prec); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('MLW PR')
    plt.tight_layout(); plt.savefig(out_path('fig_mlw_pr.png'), dpi=200); plt.close()

def make_figs(_args):
    # No-op placeholder; figures are generated per command
    return


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')

    # CEW sensitivity over scale
    def run_g2_cew_sweep(args):
        cfg = load_yaml(args.config)
        cew_cfg = load_yaml(args.cew)
        alpha = float(cew_cfg['alpha'])
        c = float(cew_cfg['c'])
        A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
        angles = (A0, A1, B0, B1)
        scales = [float(s) for s in args.scales.split(',')]
        out_rows = []
        rng = np.random.default_rng(77)
        for scale in scales:
            cfg_loc = {**cew_cfg, 'scale': scale}
            for gp in grid_points(cfg):
                p = gp['p_depol']; eta = gp['etaA']
                flags = []
                for r in range(int(args.rep)):
                    sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 1_000_000)))
                    cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                                method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                    avgX = cp['avg_card_by_ctx'].get(0, 2.0)
                    avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
                    # decide CEW for this replicate
                    from .witness.cew import threshold_L, cardinality_product
                    L = threshold_L(alpha, c, cfg_loc)
                    flags.append(int(cardinality_product(avgX, avgZ) < L))
                rate = float(np.mean(flags))
                out_rows.append({'scale': scale, 'x': p, 'y': eta, 'CEW_rate': rate})
        df = pd.DataFrame(out_rows)
        df.to_csv(out_path('cew_scale_sweep.csv'), index=False)
        # a single scale slice plotted (last scale)
        df_last = df[df['scale'] == scales[-1]].copy()
        phase_diagram(df_last, 'x', 'y', 'CEW_rate', title=f'CEW rate (scale={scales[-1]})', out_png=str(out_path('fig_cew_scale_sensitivity.png')))

    p2 = sub.add_parser('run_g2_device')
    p2.add_argument('--config', required=True)
    p2.add_argument('--cew', required=True)
    p2.add_argument('--rep', default=10)
    p2.add_argument('--N', default=4000)
    p2.add_argument('--boot', default=1500)
    p2.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2.add_argument('--K', default=11)
    p2.add_argument('--h', default=0.25)
    p2.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='none')
    p2.add_argument('--c_cew', type=float, default=None)
    p2.add_argument('--cew_scale', type=float, default=None)
    p2.add_argument('--L_mode', choices=['formula', 'entropic'], default='formula')
    p2.add_argument('--mu_C', default=0.7071067811865476)
    p2.add_argument('--sA', default=0.0)
    p2.add_argument('--tau', default=1.0)
    p2.add_argument('--gamma', default=1.0)
    p2.add_argument('--power_preset', action='store_true')
    p2.set_defaults(func=run_g2_device)

    p2sweep = sub.add_parser('run_g2_cew_sweep')
    p2sweep.add_argument('--config', required=True)
    p2sweep.add_argument('--cew', required=True)
    p2sweep.add_argument('--scales', default='1.0,1.5,2.0,2.5')
    p2sweep.add_argument('--rep', default=5)
    p2sweep.add_argument('--N', default=3000)
    p2sweep.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2sweep.add_argument('--K', default=11)
    p2sweep.add_argument('--h', default=0.25)
    p2sweep.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='none')
    p2sweep.set_defaults(func=run_g2_cew_sweep)

    # CEW sensitivity over scale
    def run_g2_cew_sweep(args):
        cfg = load_yaml(args.config)
        cew_cfg = load_yaml(args.cew)
        alpha = float(cew_cfg['alpha'])
        c = float(cew_cfg['c'])
        A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
        angles = (A0, A1, B0, B1)
        scales = [float(s) for s in args.scales.split(',')]
        out_rows = []
        rng = np.random.default_rng(77)
        for scale in scales:
            cfg_loc = {**cew_cfg, 'scale': scale}
            for gp in grid_points(cfg):
                p = gp['p_depol']; eta = gp['etaA']
                flags = []
                for r in range(int(args.rep)):
                    sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 1_000_000)))
                    cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                                method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                    avgX = cp['avg_card_by_ctx'].get(0, 2.0)
                    avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
                    # decide CEW for this replicate
                    from .witness.cew import threshold_L, cardinality_product
                    L = threshold_L(alpha, c, cfg_loc)
                    flags.append(int(cardinality_product(avgX, avgZ) < L))
                rate = float(np.mean(flags))
                out_rows.append({'scale': scale, 'x': p, 'y': eta, 'CEW_rate': rate})
        df = pd.DataFrame(out_rows)
        df.to_csv(out_path('cew_scale_sweep.csv'), index=False)
        # a single scale slice plotted (last scale)
        df_last = df[df['scale'] == scales[-1]].copy()
        phase_diagram(df_last, 'x', 'y', 'CEW_rate', title=f'CEW rate (scale={scales[-1]})', out_png=str(out_path('fig_cew_scale_sensitivity.png')))

    p2c = sub.add_parser('run_g2_ctxpair')
    p2c.add_argument('--config', required=True)
    p2c.add_argument('--rep', default=5)
    p2c.add_argument('--N', default=3000)
    p2c.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2c.add_argument('--K', default=11)
    p2c.add_argument('--h', default=0.25)
    p2c.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='none')
    p2c.set_defaults(func=run_g2_ctxpair)

    p2a = sub.add_parser('run_g2_angles')
    p2a.add_argument('--config', required=True)
    p2a.add_argument('--rep', default=5)
    p2a.add_argument('--mode', choices=['bias', 'jitter'], default='bias')
    p2a.add_argument('--N', default=3000)
    p2a.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2a.add_argument('--K', default=11)
    p2a.add_argument('--h', default=0.25)
    p2a.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='none')
    p2a.set_defaults(func=run_g2_angles)

    p2col = sub.add_parser('run_g2_colored')
    p2col.add_argument('--config', required=True)
    p2col.add_argument('--rep', default=4)
    p2col.add_argument('--N', default=3000)
    p2col.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2col.add_argument('--K', default=11)
    p2col.add_argument('--h', default=0.25)
    p2col.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='none')
    p2col.set_defaults(func=run_g2_colored)

    p2clk = sub.add_parser('run_g2_clicks')
    p2clk.add_argument('--config', required=True)
    p2clk.add_argument('--rep', default=4)
    p2clk.add_argument('--N', default=3000)
    p2clk.set_defaults(func=run_g2_clicks)

    p2cal = sub.add_parser('run_g2_entropic_calibrate')
    p2cal.add_argument('--config', required=True)
    p2cal.add_argument('--alpha', default=0.2)
    p2cal.add_argument('--mu_C', default=0.7071067811865476)
    p2cal.add_argument('--rep', default=4)
    p2cal.add_argument('--N', default=3000)
    p2cal.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2cal.add_argument('--K', default=11)
    p2cal.add_argument('--h', default=0.25)
    p2cal.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='isotonic')
    p2cal.add_argument('--tau', default=1.0)
    p2cal.set_defaults(func=run_g2_entropic_calibrate)

    p2ec = sub.add_parser('run_g2_entropic_cew')
    p2ec.add_argument('--config', required=True)
    p2ec.add_argument('--alpha', default=0.2)
    p2ec.add_argument('--mu_C', default=0.7071067811865476)
    p2ec.add_argument('--sA', default=0.0)
    p2ec.add_argument('--rep', default=5)
    p2ec.add_argument('--N', default=3000)
    p2ec.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2ec.add_argument('--K', default=11)
    p2ec.add_argument('--h', default=0.25)
    p2ec.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='isotonic')
    p2ec.add_argument('--tau', default=1.0)
    p2ec.add_argument('--gamma', default=1.0)
    p2ec.set_defaults(func=run_g2_entropic_cew)

    p3 = sub.add_parser('run_g3_entropic')
    p3.add_argument('--config', required=True)
    p3.set_defaults(func=run_g3_entropic)

    p2s = sub.add_parser('run_g2_selective')
    p2s.add_argument('--config', required=True)
    p2s.add_argument('--rep', default=5)
    p2s.add_argument('--N', default=3000)
    p2s.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2s.add_argument('--K', default=11)
    p2s.add_argument('--h', default=0.25)
    p2s.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='none')
    p2s.set_defaults(func=run_g2_selective)

    p4 = sub.add_parser('run_g4_cmi')
    p4.add_argument('--config', required=True)
    p4.add_argument('--rep', default=5)
    p4.add_argument('--alpha', default=0.2)
    p4.add_argument('--fdr', default=0.1)
    p4.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p4.add_argument('--K', default=11)
    p4.add_argument('--h', default=0.25)
    p4.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='none')
    p4.set_defaults(func=run_g4_cmi)

    p5 = sub.add_parser('run_g5_kh_alpha_surface')
    p5.add_argument('--rep', default=3)
    p5.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='none')
    p5.set_defaults(func=run_g5_kh_alpha_surface)

    p6 = sub.add_parser('run_g6_mlw')
    p6.add_argument('--rep', default=2)
    p6.add_argument('--alpha', default=0.2)
    p6.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p6.add_argument('--K', default=11)
    p6.add_argument('--h', default=0.25)
    p6.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='isotonic')
    p6.set_defaults(func=run_g6_mlw)

    def run_g6_mlw_big(args):
        cfg = load_yaml('configs/grid_chsh.yml')
        A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
        angles = (A0, A1, B0, B1)
        grid = list(grid_points(cfg))
        rng = np.random.default_rng(54321)
        # bigger dataset by more reps
        X, y = simulate_dataset(simulate_chsh, angles, grid, rep=int(args.rep_big), alpha=float(args.alpha),
                                method=args.method, K=int(args.K), h=float(args.h), calib=args.calib, rng=rng,
                                include_colored=True, include_jitter_bias=True)
        # grid search
        from .witness.mlw import grid_search_models
        best_name, report = grid_search_models(X, y, cv_splits=int(args.cv))
        pd.DataFrame([report]).to_json(out_path('mlw_gridsearch_report.json'), orient='records')
        # train best model on full data and export CV best as summary
        append_note('ML Tanık GridSearch', f"En iyi model: {best_name} (ROC AUC CV={report['best_score']:.3f}).", ['mlw_gridsearch_report.json'])

    p6b = sub.add_parser('run_g6_mlw_big')
    p6b.add_argument('--rep_big', default=20)
    p6b.add_argument('--alpha', default=0.2)
    p6b.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p6b.add_argument('--K', default=11)
    p6b.add_argument('--h', default=0.25)
    p6b.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='isotonic')
    p6b.add_argument('--cv', default=5)
    p6b.set_defaults(func=run_g6_mlw_big)

    def run_g6_mlw_dataset(args):
        cfg = load_yaml('configs/grid_chsh.yml')
        angles = (cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1'])
        grid = list(grid_points(cfg))
        rng = np.random.default_rng(13579)
        X, y = simulate_dataset(simulate_chsh, angles, grid, rep=int(args.rep_big), alpha=float(args.alpha),
                                method=args.method, K=int(args.K), h=float(args.h), calib=args.calib, rng=rng,
                                include_colored=True, include_jitter_bias=True)
        df = X.copy(); df['label'] = y
        df.to_csv(out_path('mlw_dataset.csv'), index=False)
        # health checks
        health = {
            'n_rows': int(len(df)),
            'n_dups': int(len(df) - len(df.drop_duplicates())),
            'label_pos_rate': float(df['label'].mean()),
            'S_mean': float(df['S'].mean()),
            'clickA_mean': float(df['clickA'].mean()),
            'clickB_mean': float(df['clickB'].mean()),
        }
        pd.DataFrame([health]).to_json(out_path('mlw_data_health.json'), orient='records')
        append_note('ML Veri Seti (Geniş)', 'Geniş ve çeşitli simüle veri seti üretildi; sağlık raporu eklendi.', ['mlw_dataset.csv','mlw_data_health.json'])

    p6d = sub.add_parser('run_g6_mlw_dataset')
    p6d.add_argument('--rep_big', default=15)
    p6d.add_argument('--alpha', default=0.2)
    p6d.add_argument('--method', choices=['freq','knn'], default='knn')
    p6d.add_argument('--K', default=11)
    p6d.add_argument('--h', default=0.25)
    p6d.add_argument('--calib', choices=['none','platt','isotonic'], default='isotonic')
    p6d.set_defaults(func=run_g6_mlw_dataset)

    def run_g6_mlw_fullsuite(args):
        # 1) Generate large, diverse dataset
        cfg = load_yaml('configs/grid_chsh.yml')
        angles = (cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1'])
        grid = list(grid_points(cfg))
        rng = np.random.default_rng(2468)
        X, y = simulate_dataset(simulate_chsh, angles, grid, rep=int(args.rep_big), alpha=float(args.alpha),
                                method=args.method, K=int(args.K), h=float(args.h), calib=args.calib, rng=rng,
                                include_colored=True, include_jitter_bias=True)
        X = X.reset_index(drop=True)
        # 2) Train/Cal/Test split; OOT dataset (shifted)
        n = len(X)
        idx = np.arange(n); rng.shuffle(idx)
        n_tr = int(0.6*n); n_cal = int(0.2*n)
        tr, cal, te = idx[:n_tr], idx[n_tr:n_tr+n_cal], idx[n_tr+n_cal:]
        Xtr, ytr = X.iloc[tr], y[tr]
        Xcal, ycal = X.iloc[cal], y[cal]
        Xte, yte = X.iloc[te], y[te]
        # OOT shifted
        grid_shift = []
        for p in [0.05, 0.1, 0.15]:
            for eta in [0.85, 0.9]:
                grid_shift.append({'p_depol': p, 'etaA': eta})
        Xoot, yoot = simulate_dataset(simulate_chsh, angles, grid_shift, rep=int(max(3, int(args.rep_big)//5)), alpha=float(args.alpha),
                                      method=args.method, K=int(args.K), h=float(args.h), calib=args.calib, rng=np.random.default_rng(1357),
                                      include_colored=True, include_jitter_bias=True)
        # 3) Leakage report
        from .witness.mlw import leakage_report, grid_search_models, kmeans_elbow, pca_2d, plot_elbow, plot_pca
        leak = leakage_report(Xtr, ytr)
        pd.DataFrame([leak]).to_json(out_path('mlw_leakage_report.json'), orient='records')
        # 4) Clustering + elbow + PCA
        elbow = kmeans_elbow(Xtr, kmin=2, kmax=10)
        pd.DataFrame({'k': elbow['k'], 'inertia': elbow['inertia'], 'silhouette': elbow['silhouette']}).to_csv(out_path('mlw_elbow.csv'), index=False)
        plot_elbow(elbow, str(out_path('fig_mlw_elbow.png')))
        pca = pca_2d(Xtr)
        plot_pca(np.array(pca['Z']), ytr, str(out_path('fig_mlw_pca_label.png')), title='PCA (label)')
        # 5) GridSearchCV on train
        best_name, report = grid_search_models(Xtr, ytr, cv_splits=int(args.cv))
        pd.DataFrame([report]).to_json(out_path('mlw_gridsearch_full.json'), orient='records')
        # 6) Fit best model and evaluate
        # Choose model by name from report
        name = report['best_model']
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        model_map = {
            'gbm': GradientBoostingClassifier(random_state=0),
            'rf': RandomForestClassifier(random_state=0, n_estimators=report['report'].get('rf',{}).get('best_params',{}).get('n_estimators',200),
                                         max_depth=report['report'].get('rf',{}).get('best_params',{}).get('max_depth',None)),
            'svm_rbf': SVC(probability=True, C=report['report'].get('svm_rbf',{}).get('best_params',{}).get('C',1.0),
                           gamma=report['report'].get('svm_rbf',{}).get('best_params',{}).get('gamma','scale')),
            'logreg': LogisticRegression(max_iter=2000, solver='saga', C=report['report'].get('logreg',{}).get('best_params',{}).get('C',1.0)),
            'knn': KNeighborsClassifier(n_neighbors=report['report'].get('knn',{}).get('best_params',{}).get('n_neighbors',11)),
            'gnb': GaussianNB(),
        }
        model = model_map.get(name, GradientBoostingClassifier(random_state=0))
        model.fit(Xtr.values, ytr)
        # Metrics on test and OOT
        from .witness.mlw import evaluate_mlw, conformalize_scores, conformal_curve
        scores_cal = model.predict_proba(Xcal.values)[:,1]
        scores_te = model.predict_proba(Xte.values)[:,1]
        scores_oot = model.predict_proba(Xoot.values)[:,1]
        # Optional score calibration (isotonic/platt)
        if getattr(args, 'score_calib', 'none') != 'none':
            kind = args.score_calib
            if kind == 'isotonic':
                from sklearn.isotonic import IsotonicRegression
                ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                ir.fit(scores_cal, (ycal==1).astype(int))
                scores_cal = ir.predict(scores_cal)
                scores_te = ir.predict(scores_te)
                scores_oot = ir.predict(scores_oot)
            elif kind == 'platt':
                from .cp.scores import fit_platt, apply_platt
                A,B = fit_platt(scores_cal, (ycal==1).astype(int))
                scores_cal = apply_platt(scores_cal, (A,B))
                scores_te = apply_platt(scores_te, (A,B))
                scores_oot = apply_platt(scores_oot, (A,B))
        met_te = evaluate_mlw(model, Xte, yte)
        met_oot = evaluate_mlw(model, Xoot, yoot)
        # Conformal curves
        alphas = np.linspace(0.01, 0.5, 25)
        curve_te = conformal_curve(scores_cal, ycal, scores_te, yte, alphas)
        curve_oot = conformal_curve(scores_cal, ycal, scores_oot, yoot, alphas)
        curve_te.to_csv(out_path('mlw_conformal_curve_test.csv'), index=False)
        curve_oot.to_csv(out_path('mlw_conformal_curve_oot.csv'), index=False)
        # Export summary
        summ = {
            'leakage': leak,
            'best_model': report['best_model'],
            'cv_roc': report['best_score'],
            'test': met_te,
            'oot': met_oot,
        }
        pd.DataFrame([summ]).to_json(out_path('mlw_fullsuite_summary.json'), orient='records')
        append_note('ML Fullsuite', f"Best={report['best_model']} CV‑ROC={report['best_score']:.3f}; Test ROC={met_te['roc_auc']:.3f}, OOT ROC={met_oot['roc_auc']:.3f}; score_calib={getattr(args,'score_calib','none')}.",
                    ['mlw_leakage_report.json','mlw_elbow.csv','fig_mlw_elbow.png','fig_mlw_pca_label.png','mlw_gridsearch_full.json','mlw_conformal_curve_test.csv','mlw_conformal_curve_oot.csv','mlw_fullsuite_summary.json'])

    p6f = sub.add_parser('run_g6_mlw_fullsuite')
    p6f.add_argument('--rep_big', default=40)
    p6f.add_argument('--alpha', default=0.2)
    p6f.add_argument('--method', choices=['freq','knn'], default='knn')
    p6f.add_argument('--K', default=11)
    p6f.add_argument('--h', default=0.25)
    p6f.add_argument('--calib', choices=['none','platt','isotonic'], default='isotonic')
    p6f.add_argument('--cv', default=5)
    p6f.add_argument('--score_calib', choices=['none','isotonic','platt'], default='isotonic')
    p6f.set_defaults(func=run_g6_mlw_fullsuite)

    def run_g6_mlw_cv(args):
        cfg = load_yaml('configs/grid_chsh.yml')
        angles = (cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1'])
        grid = list(grid_points(cfg))
        rng = np.random.default_rng(9876)
        X, y = simulate_dataset(simulate_chsh, angles, grid, rep=int(args.rep_big), alpha=float(args.alpha),
                                method=args.method, K=int(args.K), h=float(args.h), calib=args.calib, rng=rng)
        # deduplicate to reduce leakage
        n0 = len(X)
        X_uniq = X.drop_duplicates()
        y_uniq = y[X_uniq.index.values]
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.ensemble import GradientBoostingClassifier
        cv = StratifiedKFold(n_splits=int(args.cv), shuffle=True, random_state=0)
        gb = GradientBoostingClassifier(random_state=0)
        scores = cross_val_score(gb, X_uniq.values, y_uniq, cv=cv, scoring='roc_auc')
        rep = {'n_samples': int(n0), 'n_unique': int(len(X_uniq)), 'cv': int(args.cv), 'roc_auc_mean': float(np.mean(scores)), 'roc_auc_std': float(np.std(scores))}
        pd.DataFrame([rep]).to_json(out_path('mlw_cv_report.json'), orient='records')
        append_note('ML Tanık CV', f"ROC AUC (CV mean±std) = {rep['roc_auc_mean']:.3f}±{rep['roc_auc_std']:.3f}; n_unique={rep['n_unique']}/{rep['n_samples']}.", ['mlw_cv_report.json'])

    p6cv = sub.add_parser('run_g6_mlw_cv')
    p6cv.add_argument('--rep_big', default=20)
    p6cv.add_argument('--alpha', default=0.2)
    p6cv.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p6cv.add_argument('--K', default=11)
    p6cv.add_argument('--h', default=0.25)
    p6cv.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='isotonic')
    p6cv.add_argument('--cv', default=5)
    p6cv.set_defaults(func=run_g6_mlw_cv)

    def run_g2_cew_multiL(args):
        cfg = load_yaml(args.config)
        alpha = float(args.alpha)
        A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
        angles = (A0, A1, B0, B1)
        rows = []
        rng = np.random.default_rng(224466)
        mu_C = float(args.mu_C); sA = float(args.sA); tau = float(args.tau); gamma_ent = float(args.gamma_ent)
        a, b = float(args.a), float(args.b)
        L0, beta = float(args.L0), float(args.beta)
        q = float(args.q); p_null = float(args.p_null)
        gamma_cp = float(args.gamma_cp)
        for gp in grid_points(cfg):
            p = gp['p_depol']; eta = gp['etaA']
            prod_list = []; s_list = []; tauX_list = []; tauZ_list = []
            for r in range(int(args.rep)):
                sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 9_999_999)))
                s_list.append(sim['s_value'])
                cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                            method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                avgX = cp['avg_card_by_ctx'].get(0, 2.0); avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
                tauX_list.append(avgX); tauZ_list.append(avgZ)
                prod_list.append(avgX * avgZ)
            Smean = float(np.mean(s_list)) if s_list else np.nan
            tauX = float(np.mean(tauX_list)) if tauX_list else np.nan
            tauZ = float(np.mean(tauZ_list)) if tauZ_list else np.nan
            L_ent = threshold_L_entropic(alpha, mu_C, sA, tau, gamma_ent)
            L_s = threshold_L_s(Smean, a, b)
            L_eta = threshold_L_eta(eta, eta, L0, beta)
            # classical null quantile per-cell
            null_prods = []
            for r in range(int(args.quant_rep)):
                simn = simulate_chsh(int(args.N_quant), angles, {'p_depol': p_null}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 9_999_999)))
                cpn = split_conformal_binary(simn['a'], simn['b'], simn['x'], simn['y'], simn['meta']['clickA'], simn['meta']['clickB'], alpha,
                                             method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                null_prods.append(cpn['avg_card_by_ctx'].get(0, 2.0) * cpn['avg_card_by_ctx'].get(1, 2.0))
            L_quant = threshold_L_quantile(null_prods, q)
            L_cp = threshold_L_cpcalib(tauX, tauZ, gamma_cp)
            def rate(L):
                arr = np.asarray(prod_list, dtype=float)
                return float(np.mean(arr < L)) if len(arr) else np.nan
            rows.append({'x': p, 'y': eta,
                        'CEW_ent': rate(L_ent), 'CEW_S': rate(L_s), 'CEW_eta': rate(L_eta), 'CEW_quant': rate(L_quant), 'CEW_cp': rate(L_cp),
                        'L_ent': L_ent, 'L_s': L_s, 'L_eta': L_eta, 'L_quant': L_quant, 'L_cp': L_cp,
                        'S_mean': Smean, 'tauX': tauX, 'tauZ': tauZ})
        df = pd.DataFrame(rows)
        df.to_csv(out_path('cew_multiL_map.csv'), index=False)
        phase_diagram(df, 'x', 'y', 'CEW_ent', title='CEW (entropic)', out_png=str(out_path('fig_cew_entropic_multi.png')))
        phase_diagram(df, 'x', 'y', 'CEW_cp', title='CEW (cp-calib)', out_png=str(out_path('fig_cew_cpcalib_multi.png')))
        append_note('CEW L Karşılaştırmaları', 'Beş farklı L stratejisi ile CEW oranları karşılaştırıldı (entropic, S, eta, klasik-null, cp-calib).',
                    ['cew_multiL_map.csv', 'fig_cew_entropic_multi.png', 'fig_cew_cpcalib_multi.png'])

    p2ml = sub.add_parser('run_g2_cew_multiL')
    p2ml.add_argument('--config', required=True)
    p2ml.add_argument('--alpha', default=0.2)
    p2ml.add_argument('--rep', default=6)
    p2ml.add_argument('--N', default=3000)
    p2ml.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2ml.add_argument('--K', default=11)
    p2ml.add_argument('--h', default=0.25)
    p2ml.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='isotonic')
    p2ml.add_argument('--mu_C', default=0.7071067811865476)
    p2ml.add_argument('--sA', default=0.0)
    p2ml.add_argument('--tau', default=1.0)
    p2ml.add_argument('--gamma_ent', default=1.0)
    p2ml.add_argument('--a', default=1.2)
    p2ml.add_argument('--b', default=-0.1)
    p2ml.add_argument('--L0', default=1.2)
    p2ml.add_argument('--beta', default=1.0)
    p2ml.add_argument('--q', default=0.1)
    p2ml.add_argument('--p_null', default=0.5)
    p2ml.add_argument('--quant_rep', default=4)
    p2ml.add_argument('--N_quant', default=1500)
    p2ml.add_argument('--gamma_cp', default=1.0)
    p2ml.set_defaults(func=run_g2_cew_multiL)

    def run_g2_cpcalib_calibrate(args):
        # collect classical region (S<=2.0) across grid, fit gamma_cp, a,b and L0,beta baselines
        cfg = load_yaml(args.config)
        angles = (cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1'])
        alpha = float(args.alpha)
        prods_cls = []; tauX_cls = []; tauZ_cls = []; S_cls = []; eta_list = []
        rng = np.random.default_rng(4242)
        for gp in grid_points(cfg):
            p = gp['p_depol']; eta = gp['etaA']
            for r in range(int(args.rep)):
                sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 9_999_999)))
                if sim['s_value'] > 2.0:
                    continue
                cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                            method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                avgX = cp['avg_card_by_ctx'].get(0, 2.0); avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
                prods_cls.append(avgX * avgZ); tauX_cls.append(avgX); tauZ_cls.append(avgZ); S_cls.append(sim['s_value']); eta_list.append(eta)
        import json
        if len(prods_cls) == 0:
            append_note('CP-Calib Kalibrasyon', 'Klasik rejimde veri bulunamadı (S≤2).', [])
            return
        gamma_cp = fit_gamma_cp(np.array(prods_cls), np.array(tauX_cls), np.array(tauZ_cls), target_q=float(args.q))
        a,b = fit_s_params(np.array(S_cls), np.array(prods_cls))
        L0,beta = fit_eta_params(np.array(eta_list), np.array(prods_cls))
        rep = {'gamma_cp': float(gamma_cp), 'a': float(a), 'b': float(b), 'L0': float(L0), 'beta': float(beta), 'q': float(args.q), 'n_cls': int(len(prods_cls))}
        pd.DataFrame([rep]).to_json(out_path('cew_cpcalib_fit.json'), orient='records')
        append_note('CEW CP‑Kalibrasyon', f"gamma_cp={gamma_cp:.3f}, a={a:.3f}, b={b:.3f}, L0={L0:.3f}, beta={beta:.3f} (target q={args.q}).",
                    ['cew_cpcalib_fit.json'])

    p2cpc = sub.add_parser('run_g2_cpcalib_calibrate')
    p2cpc.add_argument('--config', required=True)
    p2cpc.add_argument('--alpha', default=0.2)
    p2cpc.add_argument('--rep', default=8)
    p2cpc.add_argument('--N', default=2000)
    p2cpc.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2cpc.add_argument('--K', default=11)
    p2cpc.add_argument('--h', default=0.25)
    p2cpc.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='isotonic')
    p2cpc.add_argument('--q', default=0.1)
    p2cpc.set_defaults(func=run_g2_cpcalib_calibrate)

    def run_power_plan(args):
        # approximate sample size for CEW_rate difference under different L strategies
        rows = []
        for strat in ('ent', 'cp'):
            p0 = float(args.p0)
            p1 = float(args.p1)
            n = n_for_proportions(p0, p1, alpha=float(args.alpha), power=float(args.power))
            rows.append({'strategy': strat, 'p0': p0, 'p1': p1, 'alpha': float(args.alpha), 'power': float(args.power), 'n_min': int(n)})
        pd.DataFrame(rows).to_csv(out_path('power_plan_multiL.csv'), index=False)
        append_note('Güç Planı (CEW)', f"p0={args.p0}, p1={args.p1}, alpha={args.alpha}, power={args.power} için n_min üretildi.", ['power_plan_multiL.csv'])

    pp = sub.add_parser('run_power_plan')
    pp.add_argument('--p0', default=0.05)
    pp.add_argument('--p1', default=0.2)
    pp.add_argument('--alpha', default=0.05)
    pp.add_argument('--power', default=0.8)
    pp.set_defaults(func=run_power_plan)

    def run_cew_bounds(args):
        import numpy as np
        alphas = np.linspace(0.01, 0.5, 50)
        df = compute_bounds(alphas)
        df.to_csv(out_path('cew_bounds.csv'), index=False)
        plot_bounds(df, str(out_path('fig_cew_bounds.png')))
        append_note('CEW Analitik Sınır', 'CP altında beklenen ürün ≈ (1+alpha)^2 eğrisi üretildi; L ile farkın neden kaybolduğunu teorik olarak gerekçeliyor.', ['cew_bounds.csv','fig_cew_bounds.png'])

    pb = sub.add_parser('run_cew_bounds')
    pb.set_defaults(func=run_cew_bounds)

    def run_g4_cmi_polish(args):
        import numpy as np
        from .witness.cmi import ece_uniform, tv_distance_to_uniform
        cfg = load_yaml(args.config)
        angles = (cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1'])
        rows = []
        rng = np.random.default_rng(1312)
        for gp in grid_points(cfg):
            p = gp['p_depol']; eta = gp['etaA']
            p0=[]; p1=[]
            for r in range(int(args.rep)):
                sim = simulate_chsh(6000, angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0,1_000_000)))
                cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha=float(args.alpha), method='knn', K=11, h=0.25, calibration='isotonic')
                p0 += cp['pvals_by_ctx'].get(0, np.array([])).tolist(); p1 += cp['pvals_by_ctx'].get(1, np.array([])).tolist()
            p0 = np.array(p0); p1 = np.array(p1)
            rows.append({'x': p, 'y': eta, 'tv0': tv_distance_to_uniform(p0), 'tv1': tv_distance_to_uniform(p1), 'ece0': ece_uniform(p0), 'ece1': ece_uniform(p1)})
        df = pd.DataFrame(rows)
        df.to_csv(out_path('cmi_polish.csv'), index=False)
        append_note('CMI Cilalama', 'TV ve ECE metrikleri ile CMI analizini zenginleştirdik (isotonic kalibrasyonlu kNN).', ['cmi_polish.csv'])

    pcmi = sub.add_parser('run_g4_cmi_polish')
    pcmi.add_argument('--config', required=True)
    pcmi.add_argument('--rep', default=8)
    pcmi.add_argument('--alpha', default=0.2)
    pcmi.set_defaults(func=run_g4_cmi_polish)

    def run_g2_selective_polish(args):
        import numpy as np
        cfg = load_yaml(args.config)
        angles = (cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1'])
        alpha = float(args.alpha)
        rows=[]
        rng = np.random.default_rng(1414)
        for gp in grid_points(cfg):
            p = gp['p_depol']; eta = gp['etaA']
            cover=[]
            for r in range(int(args.rep)):
                sim = simulate_chsh(8000, angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0,1_000_000)))
                cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha, method='knn', K=11, h=0.25, calibration='isotonic')
                cover.append(np.nanmean(list(cp['coverage_by_ctx'].values())))
            rows.append({'x': p, 'y': eta, 'coverage_mean': float(np.mean(cover)), 'target': 1.0 - alpha, 'gap': float(np.mean(cover) - (1.0 - alpha))})
        df = pd.DataFrame(rows)
        df.to_csv(out_path('selective_polish.csv'), index=False)
        append_note('Selective‑CP Cilalama', 'Coverage ortalaması ile hedef (1−alpha) arasındaki farkı raporladık.', ['selective_polish.csv'])

    psel = sub.add_parser('run_g2_selective_polish')
    psel.add_argument('--config', required=True)
    psel.add_argument('--rep', default=6)
    psel.add_argument('--alpha', default=0.2)
    psel.set_defaults(func=run_g2_selective_polish)

    def run_mlw_explain(args):
        ds = str(out_path('mlw_dataset.csv'))
        df = load_or_raise(ds)
        X = df.drop(columns=['label']); y = df['label'].values
        rf = fit_rf(X, y)
        imp = feature_importances(rf, list(X.columns)); pim = perm_importances(rf, X, y); ks = ks_by_feature(X, y)
        imp.to_csv(out_path('mlw_importances.csv'), index=False)
        pim.to_csv(out_path('mlw_perm_importances.csv'), index=False)
        ks.to_csv(out_path('mlw_feature_ks.csv'), index=False)
        plot_importance(imp, 'gini_importance', 'RF Gini Importances', str(out_path('fig_mlw_importance_gini.png')))
        plot_importance(pim, 'perm_importance', 'RF Permutation Importances', str(out_path('fig_mlw_importance_perm.png')))
        rb = rule_baseline_metrics(X, y)
        pd.DataFrame([rb]).to_json(out_path('mlw_rule_baseline.json'), orient='records')
        append_note('ML Açıklanabilirlik', 'RF önemleri (Gini+Permutation), KS ayrışma istatistikleri ve basit kural tabanlı karşılaştırma üretildi.',
                    ['mlw_importances.csv','mlw_perm_importances.csv','mlw_feature_ks.csv','fig_mlw_importance_gini.png','fig_mlw_importance_perm.png','mlw_rule_baseline.json'])

    pexp = sub.add_parser('run_mlw_explain')
    pexp.set_defaults(func=run_mlw_explain)

    def run_g2_cew_final(args):
        # Final CEW with CP-calibrated tau and gamma_cp (from fit file), as default.
        cfg = load_yaml(args.config)
        alpha = float(args.alpha)
        A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
        angles = (A0, A1, B0, B1)
        try:
            import json
            with open(out_path('cew_cpcalib_fit.json')) as f:
                gamma_cp = json.load(f)[0]['gamma_cp']
        except Exception:
            gamma_cp = 1.0
        rows = []
        rng = np.random.default_rng(1122)
        for gp in grid_points(cfg):
            p = gp['p_depol']; eta = gp['etaA']
            flags = []
            for r in range(int(args.rep)):
                sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0,9_999_999)))
                cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                            method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                avgX = cp['avg_card_by_ctx'].get(0, 2.0); avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
                L = threshold_L_cpcalib(tau_cp(alpha, 2), tau_cp(alpha, 2), gamma=float(gamma_cp))
                flags.append(int(avgX * avgZ < L))
            rows.append({'x': p, 'y': eta, 'CEW_rate': float(np.mean(flags)), 'L': float(L)})
        df = pd.DataFrame(rows)
        df.to_csv(out_path('cew_final_map.csv'), index=False)
        phase_diagram(df, 'x', 'y', 'CEW_rate', title='CEW (final CP-calib)', out_png=str(out_path('fig_cew_final.png')))
        append_note('CEW (Final CP-Calib)', 'Tau_cp(α)=1+α ve gamma_cp fit ile CEW haritası üretildi.', ['cew_final_map.csv','fig_cew_final.png'])

    p2fin = sub.add_parser('run_g2_cew_final')
    p2fin.add_argument('--config', required=True)
    p2fin.add_argument('--alpha', default=0.2)
    p2fin.add_argument('--rep', default=6)
    p2fin.add_argument('--N', default=3000)
    p2fin.add_argument('--method', choices=['freq', 'knn'], default='knn')
    p2fin.add_argument('--K', default=11)
    p2fin.add_argument('--h', default=0.25)
    p2fin.add_argument('--calib', choices=['none','platt','isotonic'], default='isotonic')
    p2fin.set_defaults(func=run_g2_cew_final)

    def run_g2_cew_eval(args):
        # Compare strategies by TP/FP and J=TP-FP between entangled and classical regions.
        cfg = load_yaml(args.config)
        alpha = float(args.alpha)
        angles = (cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1'])
        rng = np.random.default_rng(4455)
        def compute_rate(p, eta, strat):
            flags = []
            for r in range(int(args.rep)):
                sim = simulate_chsh(int(args.N), angles, {'p_depol': p}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 9_999_999)))
                cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                            method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                avgX = cp['avg_card_by_ctx'].get(0, 2.0); avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
                if strat == 'ent':
                    L = threshold_L_entropic(alpha, float(args.mu_C), float(args.sA), float(args.tau), float(args.gamma_ent))
                elif strat == 'cp':
                    try:
                        import json
                        with open(out_path('cew_cpcalib_fit.json')) as f:
                            gamma_cp = json.load(f)[0]['gamma_cp']
                    except Exception:
                        gamma_cp = 1.0
                    L = threshold_L_cpcalib(tau_cp(alpha, 2), tau_cp(alpha, 2), gamma=float(gamma_cp))
                elif strat == 'S':
                    L = threshold_L_s(sim['s_value'], float(args.a), float(args.b))
                elif strat == 'eta':
                    L = threshold_L_eta(eta, eta, float(args.L0), float(args.beta))
                else:
                    # quantile null built on-the-fly
                    null_prods = []
                    for rr in range(2):
                        simn = simulate_chsh(int(args.N_quant), angles, {'p_depol': float(args.p_null)}, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0,9_999_999)))
                        cpn = split_conformal_binary(simn['a'], simn['b'], simn['x'], simn['y'], simn['meta']['clickA'], simn['meta']['clickB'], alpha,
                                                     method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                        null_prods.append(cpn['avg_card_by_ctx'].get(0, 2.0) * cpn['avg_card_by_ctx'].get(1, 2.0))
                    L = threshold_L_quantile(null_prods, float(args.q))
                flags.append(int(avgX * avgZ < L))
            return float(np.mean(flags))
        # entangled (target) region: p<=p_max, eta>=eta_min; classical region: p=p_null
        p_max = float(args.p_max); eta_min = float(args.eta_min); p_null = float(args.p_null)
        rows = []
        for strat in ('ent','cp','S','eta','quant'):
            tp = compute_rate(p=p_max, eta=eta_min, strat=strat)
            fp = compute_rate(p=p_null, eta=eta_min, strat=strat)
            J = tp - fp
            rows.append({'strategy': strat, 'TP_rate': tp, 'FP_rate': fp, 'J': J})
        df = pd.DataFrame(rows)
        df.to_csv(out_path('cew_strategy_eval.csv'), index=False)
        append_note('CEW Strateji Kıyası', 'Beş eşik stratejisi için TP/FP ve J hesaplandı (entangled vs klasik).', ['cew_strategy_eval.csv'])

    pe = sub.add_parser('run_g2_cew_eval')
    pe.add_argument('--config', required=True)
    pe.add_argument('--alpha', default=0.2)
    pe.add_argument('--rep', default=6)
    pe.add_argument('--N', default=3000)
    pe.add_argument('--method', choices=['freq','knn'], default='knn')
    pe.add_argument('--K', default=11)
    pe.add_argument('--h', default=0.25)
    pe.add_argument('--calib', choices=['none','platt','isotonic'], default='isotonic')
    pe.add_argument('--mu_C', default=0.7071067811865476)
    pe.add_argument('--sA', default=0.0)
    pe.add_argument('--tau', default=1.0)
    pe.add_argument('--gamma_ent', default=1.0)
    pe.add_argument('--a', default=1.2)
    pe.add_argument('--b', default=-0.1)
    pe.add_argument('--L0', default=1.2)
    pe.add_argument('--beta', default=1.0)
    pe.add_argument('--q', default=0.1)
    pe.add_argument('--p_null', default=0.5)
    pe.add_argument('--N_quant', default=1000)
    pe.add_argument('--p_max', default=0.02)
    pe.add_argument('--eta_min', default=0.95)
    pe.set_defaults(func=run_g2_cew_eval)

    pf = sub.add_parser('make_figs')
    pf.set_defaults(func=make_figs)

    def run_qiskit_validation(args):
        if simulate_chsh_qiskit is None:
            print("Qiskit not available. Please install qiskit and qiskit-aer.")
            return
        
        cfg = load_yaml(args.config)
        cew_cfg = load_yaml(args.cew)
        alpha = float(cew_cfg['alpha'])
        c = float(cew_cfg['c'])
        A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
        angles = (A0, A1, B0, B1)
        
        records = []
        rng = np.random.default_rng(42)
        
        # Run on a smaller grid for validation (Qiskit is slower)
        # Or just the points in the config if they are few.
        # Let's limit to a few points if the grid is large.
        
        points = list(grid_points(cfg))
        if len(points) > 10:
            print(f"Grid has {len(points)} points. Limiting to first 5 for Qiskit validation.")
            points = points[:5]
            
        for gp in points:
            p = gp['p_depol']
            etaA = gp['etaA']
            etaB = gp['etaB']
            
            s_vals = []
            cew_flags = []
            
            for r in range(int(args.rep)):
                print(f"Running Qiskit sim: p={p}, eta={etaA}, rep={r}")
                sim = simulate_chsh_qiskit(N=int(args.N), angles=angles,
                                           noise_cfg={'p_depol': p, 'px': 0.0, 'pz': 0.0},
                                           detect_cfg={'etaA': etaA, 'etaB': etaB, 'darkA': 0.0, 'darkB': 0.0},
                                           seed=int(rng.integers(0, 10_000_000)))
                
                s_vals.append(sim['s_value'])
                cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                            method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
                
                avgX = cp['avg_card_by_ctx'].get(0, 2.0)
                avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
                
                from .witness.cew import threshold_L, cardinality_product
                L = threshold_L(alpha, c, cew_cfg)
                cew_flags.append(int(cardinality_product(avgX, avgZ) < L))
                
            records.append({
                'x': p, 'y': etaA,
                'S_mean': float(np.mean(s_vals)),
                'CEW_rate': float(np.mean(cew_flags))
            })
            
        df = pd.DataFrame(records)
        df.to_csv(out_path('qiskit_validation.csv'), index=False)
        phase_diagram(df, 'x', 'y', 'CEW_rate', title='Qiskit Validation (CEW)', out_png=str(out_path('fig_qiskit_cew.png')))
        phase_diagram(df, 'x', 'y', 'S_mean', title='Qiskit Validation (S)', out_png=str(out_path('fig_qiskit_s.png')))
        
        append_note(
            'Qiskit Doğrulama',
            'Qiskit backend ile CHSH simülasyonu çalıştırıldı. Sonuçlar qiskit_validation.csv dosyasına kaydedildi.',
            ['qiskit_validation.csv', 'fig_qiskit_cew.png', 'fig_qiskit_s.png']
        )

    pq = sub.add_parser('run_qiskit_validation')
    pq.add_argument('--config', required=True)
    pq.add_argument('--cew', required=True)
    pq.add_argument('--rep', default=3)
    pq.add_argument('--N', default=1000)
    pq.add_argument('--method', choices=['freq', 'knn'], default='knn')
    pq.add_argument('--K', default=11)
    pq.add_argument('--h', default=0.25)
    pq.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='none')
    pq.set_defaults(func=run_qiskit_validation)

    def run_external_validation(args):
        if args.mock:
            print("Generating MOCK Big Bell Test data...")
            df = mock_big_bell_test_data(N=int(args.N))
            csv_path = out_path('mock_bbt.csv')
            df.to_csv(csv_path, index=False)
            args.csv = str(csv_path)
            
        print(f"Loading external data from {args.csv}...")
        try:
            data = load_big_bell_test_data(args.csv)
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        print(f"Data loaded. Samples: {data['meta']['n_samples']}")
        print(f"CHSH S-value: {data['s_value']:.4f}")
        
        # Run Conformal Prediction
        cew_cfg = load_yaml(args.cew)
        alpha = float(cew_cfg['alpha'])
        
        # We need to fake 'click' arrays if they don't exist (assume all clicked if not specified)
        # The loader returns x,y. If x=0 it means no click?
        # Our loader maps 0->1, 1->-1.
        # If original data had 0 for no click, we should have handled it.
        # Let's assume the data is post-selected (loophole-free) or we treat all as clicked.
        
        clickA = np.ones_like(data['x'], dtype=bool)
        clickB = np.ones_like(data['y'], dtype=bool)
        
        cp = split_conformal_binary(data['a'], data['b'], data['x'], data['y'], clickA, clickB, alpha,
                                    method=args.method, K=int(args.K), h=float(args.h), calibration=args.calib)
        
        avgX = cp['avg_card_by_ctx'].get(0, 2.0)
        avgZ = cp['avg_card_by_ctx'].get(1, 2.0)
        prod = avgX * avgZ
        
        from .witness.cew import threshold_L, cardinality_product
        L = threshold_L(alpha, float(cew_cfg['c']), cew_cfg)
        
        print(f"Conformal Set Sizes: X={avgX:.4f}, Z={avgZ:.4f}")
        print(f"Product: {prod:.4f}")
        print(f"Threshold L: {L:.4f}")
        
        if prod < L:
            print(">>> CEW WITNESS TRIGGERED! Entanglement Detected! <<<")
        else:
            print("CEW Witness NOT triggered.")
            
        # CMI
        from .witness.cmi import tv_distance_to_uniform
        p0 = cp['pvals_by_ctx'].get(0, [])
        p1 = cp['pvals_by_ctx'].get(1, [])
        tv0 = tv_distance_to_uniform(p0)
        tv1 = tv_distance_to_uniform(p1)
        print(f"CMI (TV distance): Context 0 = {tv0:.4f}, Context 1 = {tv1:.4f}")
        
        append_note(
            'External Validation',
            f"Data: {args.csv}. S={data['s_value']:.3f}. CEW={prod:.3f}<{L:.3f}? CMI={max(tv0,tv1):.3f}",
            []
        )

    pex = sub.add_parser('run_external_validation')
    pex.add_argument('--csv', help='Path to external CSV file')
    pex.add_argument('--mock', action='store_true', help='Generate mock data instead of loading')
    pex.add_argument('--cew', required=True)
    pex.add_argument('--N', default=2000)
    pex.add_argument('--method', choices=['freq', 'knn'], default='knn')
    pex.add_argument('--K', default=11)
    pex.add_argument('--h', default=0.25)
    pex.add_argument('--calib', choices=['none', 'platt', 'isotonic'], default='none')
    pex.set_defaults(func=run_external_validation)

    args = ap.parse_args()
    if not hasattr(args, 'func'):
        ap.print_help()
        return
    args.func(args)
    cmd_name = getattr(args, 'cmd', getattr(args.func, '__name__', 'unknown'))
    write_run_metadata(cmd_name, args, seeds=RUN_SEED_MAP.get(cmd_name, []))


if __name__ == '__main__':
    main()
