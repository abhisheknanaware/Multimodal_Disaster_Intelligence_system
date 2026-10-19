"""
Trains and evaluates the v5 landslide and flood classifiers.

Evaluation is out-of-fold with spatial blocks (0.1 deg cells) so nearby points never
sit on both sides of a split; flood is also checked with leave-one-date-out CV.
The decision threshold is tuned for macro-F1 inside each training fold only,
then applied to the held-out fold.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python-ml-service'))
import feature_builder as fb  # noqa: E402
from feature_builder import LANDSLIDE_FEATURES, FLOOD_FEATURES, TRIGGER_FEATURES  # noqa: E402

DATA_DIR = os.path.join(ROOT, 'data', 'v5')
MODEL_DIR = os.path.join(ROOT, 'models', 'v5')
os.makedirs(MODEL_DIR, exist_ok=True)

CONFIGS = [
    {'depth': 4, 'l2_leaf_reg': 3, 'learning_rate': 0.05, 'iterations': 600},
    {'depth': 6, 'l2_leaf_reg': 5, 'learning_rate': 0.03, 'iterations': 800},
    {'depth': 5, 'l2_leaf_reg': 10, 'learning_rate': 0.05, 'iterations': 500},
]


def make_model(cfg):
    return CatBoostClassifier(**cfg, loss_function='Logloss', random_seed=42, verbose=False,
                              allow_writing_files=False)


def best_threshold(y, p):
    grid = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y, p >= t, average='macro') for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'roc_auc': round(roc_auc_score(y, p), 4),
        'pr_auc': round(average_precision_score(y, p), 4),
        'threshold': round(t, 3),
        'macro_f1': round(f1_score(y, pred, average='macro'), 4),
        'precision': round(precision_score(y, pred, zero_division=0), 4),
        'recall': round(recall_score(y, pred, zero_division=0), 4),
        'f1_pos': round(f1_score(y, pred, zero_division=0), 4),
        'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }


def oof_predict(X, y, groups, splitter, cfg, inner_groups=None, tune_mask=None):
    """
    Out-of-fold probabilities and per-fold thresholds tuned on the training fold only.
    `tune_mask` limits threshold tuning to real labelled rows (augmented rows still train).
    """
    p = np.zeros(len(y))
    t = np.zeros(len(y))
    tune_mask = np.ones(len(y), bool) if tune_mask is None else tune_mask
    for tr, te in splitter.split(X, y, groups):
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1)
        g_in = (inner_groups if inner_groups is not None else groups)[tr]
        p_in = np.zeros(len(tr))
        for itr, ite in inner.split(X.iloc[tr], y[tr], g_in):
            m = make_model(cfg).fit(X.iloc[tr[itr]], y[tr[itr]])
            p_in[ite] = m.predict_proba(X.iloc[tr[ite]])[:, 1]
        keep = tune_mask[tr]
        thr = best_threshold(y[tr][keep], p_in[keep])
        m = make_model(cfg).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
        t[te] = thr
    return p, t


def fold_metrics(y, p, t):
    pred = (p >= t).astype(int)
    out = metrics(y, p, float(np.median(t)))
    out['macro_f1'] = round(f1_score(y, pred, average='macro'), 4)
    out['precision'] = round(precision_score(y, pred, zero_division=0), 4)
    out['recall'] = round(recall_score(y, pred, zero_division=0), 4)
    out['f1_pos'] = round(f1_score(y, pred, zero_division=0), 4)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out['confusion'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _platt(y, p):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * _logit(p) + b)))


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean()
                     for k in range(bins) if (idx == k).any()))


def calibrate_and_levels(y, p, alert_precision):
    """
    Platt-scales out-of-fold scores and picks two operating points:
    'watch' keeps recall >= 0.90; 'alert' is the lowest threshold reaching
    `alert_precision` (or the best F0.5 if that precision is never reached).
    Calibration quality is reported with 5-fold cross-fitting.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import brier_score_loss, fbeta_score
    cross = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(p, y):
        cross[te] = apply_platt(p[te], *_platt(y[tr], p[tr]))
    a, b = _platt(y, p)
    pc = apply_platt(p, a, b)

    grid = np.unique(np.round(pc, 4))
    rec = np.array([recall_score(y, pc >= t, zero_division=0) for t in grid])
    prec = np.array([precision_score(y, pc >= t, zero_division=0) for t in grid])
    watch = float(grid[rec >= 0.90].max()) if (rec >= 0.90).any() else float(grid.min())
    usable = rec >= 0.20
    ok = (prec >= alert_precision) & usable
    if ok.any():
        alert = float(grid[ok].min())
    else:
        f05 = np.array([fbeta_score(y, pc >= t, beta=0.5, zero_division=0) for t in grid])
        alert = float(grid[int(np.argmax(np.where(usable, f05, -1)))])

    def level(t):
        m = metrics(y, pc, t)
        return {k: m[k] for k in ('threshold', 'precision', 'recall', 'macro_f1', 'confusion')}

    return {
        'calibration': {'a': a, 'b': b,
                        'brier_raw': round(brier_score_loss(y, p), 4),
                        'brier_calibrated_crossfit': round(brier_score_loss(y, cross), 4),
                        'ece_raw': round(_ece(y, p), 4),
                        'ece_calibrated_crossfit': round(_ece(y, cross), 4)},
        'alert_levels': {'watch': level(watch), 'alert': level(alert), 'alert_precision_target': alert_precision},
    }


def spatial_groups(df, cell=0.1):
    return (np.floor(df.latitude / cell).astype(int).astype(str) + '_' +
            np.floor(df.longitude / cell).astype(int).astype(str)).values


def select_config(X, y, groups):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    best = None
    for cfg in CONFIGS:
        p = np.zeros(len(y))
        for tr, te in splitter.split(X, y, groups):
            p[te] = make_model(cfg).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        score = average_precision_score(y, p)
        print(f"    cfg {cfg} -> PR-AUC {score:.4f}")
        if best is None or score > best[0]:
            best = (score, cfg)
    return best[1]


def old_landslide_scores(df):
    import catboost
    m = catboost.CatBoostRegressor()
    m.load_model(os.path.join(ROOT, 'models', 'landslide_v4_advanced.cbm'))
    slope = df.slope.fillna(0)
    feats = pd.DataFrame({
        'slope': df.slope, 'elevation': df.elevation, 'tpi': df.tpi, 'tri': df.tri,
        'aspect': np.degrees(np.arctan2(df.eastness, df.northness)) % 360,
        'sar_change': np.nan, 'rainfall': 0.0, 's2_ndvi': np.nan, 's2_ndwi': np.nan,
        'northness': df.northness, 'eastness': df.eastness,
        'twi_proxy': -df.tpi / np.tan(np.maximum(0.001, np.radians(slope))),
        'erosion_potential': 0.0,
    })[m.feature_names_]
    return np.clip(m.predict(feats), 0, 1)


def train_landslide():
    print("\n=== LANDSLIDE ===")
    df = pd.read_csv(os.path.join(DATA_DIR, 'landslide_v5.csv'))
    X, y = df[LANDSLIDE_FEATURES], df['class'].values
    groups = spatial_groups(df)
    cfg = select_config(X, y, groups)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    p, t = oof_predict(X, y, groups, splitter, cfg)
    report = {'config': cfg, 'n': int(len(df)), 'spatial_cv': fold_metrics(y, p, t)}

    hard = (df.neg_type != 'random').values
    report['spatial_cv_hard_negatives_only'] = fold_metrics(y[hard], p[hard], t[hard])

    no_road = [f for f in LANDSLIDE_FEATURES if f != 'dist_road_m']
    p_nr, t_nr = oof_predict(df[no_road], y, groups, splitter, cfg)
    report['ablation_without_road_distance'] = fold_metrics(y, p_nr, t_nr)

    old = old_landslide_scores(df)
    report['old_v4_same_points'] = metrics(y, old, best_threshold(y, old))
    report['old_v4_hard_negatives_only'] = metrics(y[hard], old[hard], best_threshold(y[hard], old[hard]))

    final = make_model(cfg).fit(X, y)
    threshold = float(np.median(t))
    final.save_model(os.path.join(MODEL_DIR, 'landslide_v5.cbm'))
    report['deploy_threshold'] = threshold
    report.update(calibrate_and_levels(y, p, alert_precision=0.80))
    report['features'] = LANDSLIDE_FEATURES
    report['feature_importance'] = dict(sorted(
        zip(LANDSLIDE_FEATURES, map(float, final.get_feature_importance())), key=lambda kv: -kv[1]))
    return report


def old_flood_scores(df):
    import catboost
    v3 = pd.read_csv(os.path.join(ROOT, 'data', 'v3', 'flood_features_v3_advanced.csv'))
    m = catboost.CatBoostClassifier()
    m.load_model(os.path.join(ROOT, 'models', 'flood_catboost_v3.cbm'))
    live = v3[m.feature_names_].copy()
    for c in ['S1_VV_change', 'S1_VH_change']:
        live[c] = 0.0  # what the live API actually feeds the old model
    return m.predict_proba(v3[m.feature_names_])[:, 1], m.predict_proba(live)[:, 1]


def load_flood_data():
    """2025 SAR labels (+ dry-season augmentation) plus 2019-2024 Sentinel-1 RTC labels if built."""
    base = pd.read_csv(os.path.join(DATA_DIR, 'flood_v5.csv'))
    base['source'] = np.where(base.augmented == 1, 'DRY_AUG', 'SAR_2025')
    frames = [base]
    s1_path = os.path.join(DATA_DIR, 'flood_s1_2019_2024.csv')
    if os.path.exists(s1_path):
        s1 = pd.read_csv(s1_path)
        # Keep every flooded observation; subsample the ~49k dry ones per tile for training speed
        rs = np.random.default_rng(5)
        pos = s1[s1.flood_label == 1]
        mon = s1[(s1.flood_label == 0) & (s1.season == 'monsoon')]
        dry = s1[s1.season == 'dry']
        mon = mon.groupby('tile', group_keys=False)[mon.columns].apply(
            lambda g: g.sample(min(len(g), int(8000 * len(g) / len(mon)) + 1), random_state=int(rs.integers(1e9))))
        dry = dry.groupby('tile', group_keys=False)[dry.columns].apply(
            lambda g: g.sample(min(len(g), int(3000 * len(g) / len(dry)) + 1), random_state=int(rs.integers(1e9))))
        s1 = pd.concat([pos, mon, dry])
        s1['source'] = np.where(s1.season == 'dry', 'S1_DRY_2019_2024', 'S1_2019_2024')
        frames.append(s1)
    keep = FLOOD_FEATURES + ['flood_label', 'augmented', 'latitude', 'longitude', 'event_date', 'source']
    df = pd.concat([f[keep] for f in frames], ignore_index=True)
    df['year'] = pd.to_datetime(df.event_date).dt.year
    return df


def train_flood():
    print("\n=== FLOOD ===")
    df = load_flood_data()
    X, y = df[FLOOD_FEATURES], df['flood_label'].values
    real = (df.augmented == 0).values
    is_2025 = (df.source == 'SAR_2025').values
    is_s1 = df.source.str.startswith('S1').values
    groups = spatial_groups(df)
    base_cfg = select_config(X, y, groups)
    # More rain or wetter soil must never lower flood risk, including beyond the training range
    cfg = {**base_cfg, 'monotone_constraints': {f: 1 for f in FLOOD_FEATURES
                                                if f.startswith('rain_')
                                                or f == 'soil_moisture'}}

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    p, t = oof_predict(X, y, groups, splitter, cfg, tune_mask=real)
    report = {'config': cfg, 'n_labelled': int(real.sum()), 'n_dry_season_aug': int((~real).sum()),
              'rows_by_source': {f'{k[0]}|{k[1]}': int(v)
                                 for k, v in df.groupby(['source', 'flood_label']).size().items()},
              'spatial_cv': fold_metrics(y[real], p[real], t[real]),
              'spatial_cv_2025_rows_only': fold_metrics(y[is_2025], p[is_2025], t[is_2025]),
              'dry_season_false_alarm_rate': round(float(np.mean(p[~real] >= t[~real])), 4)}

    if is_s1.any():
        year_groups = np.where(real, df.year.astype(str).values, 'AUG')
        p_y, t_y = oof_predict(X, y, year_groups, LeaveOneGroupOut(), cfg, inner_groups=groups, tune_mask=real)
        report['leave_one_year_out_cv'] = fold_metrics(y[real], p_y[real], t_y[real])

        # External check: learn only from the 2019-2024 Sentinel-1 labels, test on the 2025 labels
        tr = is_s1 | ~real
        p_tr, t_tr = oof_predict(X[tr], y[tr], groups[tr], splitter, cfg, tune_mask=real[tr])
        ext = make_model(cfg).fit(X[tr], y[tr]).predict_proba(X[is_2025])[:, 1]
        report['external_train_2019_2024_test_2025'] = metrics(y[is_2025], ext, float(np.median(t_tr[real[tr]])))
    else:
        date_groups = np.where(real, df.event_date.values, 'DRY')
        p_d, t_d = oof_predict(X, y, date_groups, LeaveOneGroupOut(), cfg, inner_groups=groups, tune_mask=real)
        report['leave_one_date_out_cv'] = fold_metrics(y[real], p_d[real], t_d[real])

    # Old v3 feature sets, on the 2025 rows they were built for
    v3 = pd.read_csv(os.path.join(ROOT, 'data', 'v3', 'flood_features_v3_advanced.csv'))
    old_cols = ['DEM', 'slope', 'rain_intensity', 'rainfall_1d', 'rainfall_3d', 'rainfall_7d',
                'antecedent_rainfall', 'S1_VV', 'S1_VH', 'S1_VV_change', 'S1_VH_change', 'S1_VV_VH_ratio']
    y_r, g_r = y[is_2025], groups[is_2025]
    p_o, t_o = oof_predict(v3[old_cols], y_r, g_r, splitter, base_cfg)
    report['old_v3_features_spatial_cv'] = fold_metrics(y_r, p_o, t_o)
    p_o2, t_o2 = oof_predict(v3[old_cols[:7]], y_r, g_r, splitter, base_cfg)
    report['old_v3_features_no_s1_spatial_cv'] = fold_metrics(y_r, p_o2, t_o2)

    final = make_model(cfg).fit(X, y)
    final.save_model(os.path.join(MODEL_DIR, 'flood_v5.cbm'))
    report['deploy_threshold'] = float(np.median(t[real]))
    report.update(calibrate_and_levels(y[real], p[real], alert_precision=0.85))
    report['features'] = FLOOD_FEATURES
    report['feature_importance'] = dict(sorted(
        zip(FLOOD_FEATURES, map(float, final.get_feature_importance())), key=lambda kv: -kv[1]))
    return report


def trigger_matrix(df, source='era5'):
    """
    source='era5': Open-Meteo archive windows as stored.
    source='imd':  IMD gauge-gridded windows.
    source='era5_qm': archive windows mapped to the IMD scale (what the live API feeds).
    """
    rows = []
    for r in df.itertuples():
        if source == 'imd':
            rain = {'24h_mm': r.imd_rain_1d, '3d_mm': r.imd_rain_3d, '7d_mm': r.imd_rain_7d,
                    '14d_mm': r.imd_rain_14d, '30d_mm': r.imd_rain_30d, 'max_daily_7d_mm': r.imd_max_1d_7d}
        else:
            rain = {'24h_mm': r.rain_1d, '3d_mm': r.rain_3d, '7d_mm': r.rain_7d, '14d_mm': r.rain_14d,
                    '30d_mm': r.rain_30d, 'max_daily_7d_mm': r.rain_max_1d_7d}
            if source == 'era5_qm':
                rain = fb.to_imd_scale(rain)
        rain['hours_24h'] = r.rain_hours_1d
        rows.append(fb.trigger_features(rain, r.soil_moisture, {'rain_clim_jjas': r.rain_clim_jjas}))
    return pd.DataFrame(rows)[TRIGGER_FEATURES]


def train_trigger_imd(cfg):
    """IMD-trained trigger, evaluated on IMD and on quantile-mapped archive rain (the live path)."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'landslide_trigger_v5.csv'))
    df = df[df.imd_rain_1d.notna()].reset_index(drop=True)
    y, years = df['label'].values, df.year.astype(str).values
    X_imd, X_qm, X_era5 = trigger_matrix(df, 'imd'), trigger_matrix(df, 'era5_qm'), trigger_matrix(df, 'era5')
    out = {'n_events': int(y.sum()), 'n_controls': int((y == 0).sum())}
    p_i, p_q, p_e = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X_imd, y, years):
        m_i = make_model(cfg).fit(X_imd.iloc[tr], y[tr])
        p_i[te] = m_i.predict_proba(X_imd.iloc[te])[:, 1]
        p_q[te] = m_i.predict_proba(X_qm.iloc[te])[:, 1]
        p_e[te] = make_model(cfg).fit(X_era5.iloc[tr], y[tr]).predict_proba(X_era5.iloc[te])[:, 1]
    for name, p in (('imd_train_imd_test', p_i), ('imd_train_live_qm_test', p_q), ('era5_train_era5_test', p_e)):
        out[name] = metrics(y, p, best_threshold(y, p))
    return out, make_model(cfg).fit(X_imd, y), y, p_q


def train_trigger():
    print("\n=== LANDSLIDE RAINFALL TRIGGER ===")
    df = pd.read_csv(os.path.join(DATA_DIR, 'landslide_trigger_v5.csv'))
    X, y = trigger_matrix(df), df['label'].values
    years = df.year.astype(str).values
    mono = {f: 1 for f in TRIGGER_FEATURES if f not in ('rain_3d_share_14d', 'rain_clim_jjas')}
    cfg = {'depth': 4, 'l2_leaf_reg': 5, 'learning_rate': 0.03, 'iterations': 600, 'monotone_constraints': mono}

    p, t = oof_predict(X, y, years, LeaveOneGroupOut(), cfg, inner_groups=df.cell.values)
    report = {'config': {k: v for k, v in cfg.items() if k != 'monotone_constraints'},
              'n_events': int(y.sum()), 'n_controls': int((y == 0).sum()),
              'events_by_source': df[df.label == 1].source.value_counts().to_dict(),
              'leave_one_year_out_cv': fold_metrics(y, p, t)}

    rule = np.array([fb.landslide_rain_trigger(a, b) for a, b in zip(df.rain_1d, df.rain_3d)])
    report['old_imd_rule_same_rows'] = metrics(y, rule, best_threshold(y, rule))

    final = make_model(cfg).fit(X, y)
    report['deploy_threshold'] = float(np.median(t))

    # Compare against an IMD-trained trigger on the rows IMD covers, scored the way the live
    # API would score it (archive rain mapped to the IMD scale). Deploy whichever is better.
    imd_report, imd_model, y_imd, p_live = train_trigger_imd(cfg)
    report['imd_comparison'] = imd_report
    use_imd = imd_report['imd_train_live_qm_test']['pr_auc'] > imd_report['era5_train_era5_test']['pr_auc']
    report['rain_scale'] = 'imd' if use_imd else 'era5'
    if use_imd:
        final, y_cal, p_cal = imd_model, y_imd, p_live
        X_scale = trigger_matrix(pd.read_csv(os.path.join(DATA_DIR, 'landslide_trigger_v5.csv'))
                                 .query('imd_rain_1d == imd_rain_1d').reset_index(drop=True), 'imd')
        y_scale = y_imd
    else:
        y_cal, p_cal, X_scale, y_scale = y, p, X, y
    final.save_model(os.path.join(MODEL_DIR, 'landslide_trigger_v5.cbm'))
    report.update(calibrate_and_levels(y_cal, p_cal, alert_precision=0.50))
    # Case-control sampling (1:10) makes raw scores relative; 1.0 = a typical intense landslide day
    report['trigger_scale'] = float(np.percentile(final.predict_proba(X_scale[y_scale == 1])[:, 1], 90))
    report['features'] = TRIGGER_FEATURES
    report['feature_importance'] = dict(sorted(
        zip(TRIGGER_FEATURES, map(float, final.get_feature_importance())), key=lambda kv: -kv[1]))
    return report


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out_path = os.path.join(MODEL_DIR, 'metrics_v5.json')
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    if which in ('all', 'landslide'):
        results['landslide'] = train_landslide()
    if which in ('all', 'flood'):
        results['flood'] = train_flood()
    if which in ('all', 'trigger'):
        results['landslide_trigger'] = train_trigger()
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, out_path)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('feature_importance', 'features')}
                      for k, v in results.items()}, indent=2))
