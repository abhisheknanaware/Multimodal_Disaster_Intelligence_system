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


def train_flood():
    print("\n=== FLOOD ===")
    df = pd.read_csv(os.path.join(DATA_DIR, 'flood_v5.csv'))
    X, y = df[FLOOD_FEATURES], df['flood_label'].values
    real = (df.augmented == 0).values
    groups = spatial_groups(df)
    base_cfg = select_config(X, y, groups)
    # More rain or wetter soil must never lower flood risk, including beyond the training range
    cfg = {**base_cfg, 'monotone_constraints': {f: 1 for f in FLOOD_FEATURES
                                                if f.startswith('rain_')
                                                or f == 'soil_moisture'}}

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    p, t = oof_predict(X, y, groups, splitter, cfg, tune_mask=real)
    report = {'config': cfg, 'n_labelled': int(real.sum()), 'n_dry_season_aug': int((~real).sum()),
              'spatial_cv': fold_metrics(y[real], p[real], t[real]),
              'dry_season_false_alarm_rate': round(float(np.mean(p[~real] >= t[~real])), 4)}

    date_groups = np.where(real, df.event_date.values, 'DRY')
    p_d, t_d = oof_predict(X, y, date_groups, LeaveOneGroupOut(), cfg, inner_groups=groups, tune_mask=real)
    report['leave_one_date_out_cv'] = fold_metrics(y[real], p_d[real], t_d[real])

    terrain_only = [f for f in FLOOD_FEATURES if not f.startswith('rain') and f != 'soil_moisture']
    p_s, t_s = oof_predict(df[terrain_only], y, groups, splitter, base_cfg, tune_mask=real)
    report['ablation_terrain_only_spatial_cv'] = fold_metrics(y[real], p_s[real], t_s[real])

    v3 = pd.read_csv(os.path.join(ROOT, 'data', 'v3', 'flood_features_v3_advanced.csv'))
    old_cols = ['DEM', 'slope', 'rain_intensity', 'rainfall_1d', 'rainfall_3d', 'rainfall_7d',
                'antecedent_rainfall', 'S1_VV', 'S1_VH', 'S1_VV_change', 'S1_VH_change', 'S1_VV_VH_ratio']
    y_r, g_r = y[real], groups[real]
    p_o, t_o = oof_predict(v3[old_cols], y_r, g_r, splitter, base_cfg)
    report['old_v3_features_spatial_cv'] = fold_metrics(y_r, p_o, t_o)
    p_o2, t_o2 = oof_predict(v3[old_cols[:7]], y_r, g_r, splitter, base_cfg)
    report['old_v3_features_no_s1_spatial_cv'] = fold_metrics(y_r, p_o2, t_o2)

    old_in, old_live = old_flood_scores(df)
    y = y_r
    report['old_v3_in_sample'] = metrics(y, old_in, 0.5)
    report['old_v3_as_served_live'] = metrics(y, old_live, 0.5)

    final = make_model(cfg).fit(X, df['flood_label'].values)
    final.save_model(os.path.join(MODEL_DIR, 'flood_v5.cbm'))
    report['deploy_threshold'] = float(np.median(t[real]))
    report['features'] = FLOOD_FEATURES
    report['feature_importance'] = dict(sorted(
        zip(FLOOD_FEATURES, map(float, final.get_feature_importance())), key=lambda kv: -kv[1]))
    return report


def trigger_matrix(df):
    rows = []
    for r in df.itertuples():
        rain = {'24h_mm': r.rain_1d, '3d_mm': r.rain_3d, '7d_mm': r.rain_7d, '14d_mm': r.rain_14d,
                '30d_mm': r.rain_30d, 'hours_24h': r.rain_hours_1d, 'max_daily_7d_mm': r.rain_max_1d_7d}
        rows.append(fb.trigger_features(rain, r.soil_moisture, {'rain_clim_jjas': r.rain_clim_jjas}))
    return pd.DataFrame(rows)[TRIGGER_FEATURES]


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
    final.save_model(os.path.join(MODEL_DIR, 'landslide_trigger_v5.cbm'))
    report['deploy_threshold'] = float(np.median(t))
    # Case-control sampling (1:10) makes raw scores relative; 1.0 = a typical intense landslide day
    report['trigger_scale'] = float(np.percentile(final.predict_proba(X[y == 1])[:, 1], 90))
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
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('feature_importance', 'features')}
                      for k, v in results.items()}, indent=2))
