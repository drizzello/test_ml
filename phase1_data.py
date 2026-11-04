"""Fase 1: parsing, feature, split per paziente, scorciatoia inserita, tabella riassuntiva.

Uso: uv run python phase1_data.py
Output: data/hourly.parquet, data/features.npz, data/features_meta.json,
        results/phase1.json, results/phase1.md
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import (BC, COUNT, DATA, LAST, NORMAL, RES, SEED, STATIC, TIMEVAR, ALL_FEATURES,
                    DESCRIPTION, WINDOW, write_json)

RAW = DATA / "raw"
# Probabilita' della scorciatoia inserita
P_BC_A_POS, P_BC_A_NEG, P_BC_B = 0.70, 0.05, 0.10
BC_SEED = 2019


def read_one(path):
    df = pd.read_csv(path, sep="|")
    df.insert(0, "pid", path.stem)
    return df


def parse():
    out = DATA / "hourly.parquet"
    if out.exists():
        return pd.read_parquet(out)
    frames = []
    for hosp, sub in [("A", "training_setA"), ("B", "training_setB")]:
        files = sorted((RAW / sub).glob("*.psv"))
        with Pool(4) as p:
            dfs = p.map(read_one, files, chunksize=200)
        df = pd.concat(dfs, ignore_index=True)
        df.insert(1, "hospital", hosp)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out)
    return df


def split_patients(df):
    """A: 60/20/20 train/val/test per paziente (stratificato su sepsi si/no). B: tutto test esterno."""
    rng = np.random.default_rng(SEED)
    pat = df.groupby("pid").agg(hospital=("hospital", "first"), sepsis=("SepsisLabel", "max"))
    split = pd.Series("B_test", index=pat.index)
    for s in (0, 1):
        ids = pat.index[(pat.hospital == "A") & (pat.sepsis == s)].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n_tr, n_va = int(0.6 * len(ids)), int(0.2 * len(ids))
        split[ids[:n_tr]] = "A_train"
        split[ids[n_tr:n_tr + n_va]] = "A_val"
        split[ids[n_tr + n_va:]] = "A_test"
    return df.pid.map(split).to_numpy()


def build(df, split):
    g = df.groupby("pid", sort=False)
    # numero di misurazioni nelle ultime 6 ore (ora corrente inclusa)
    notna = df[TIMEVAR].notna().astype(np.int32)
    notna["pid"] = df.pid
    cs = notna.groupby("pid", sort=False).cumsum()
    lag = cs.groupby(df.pid, sort=False).shift(WINDOW).fillna(0)
    counts = (cs - lag).to_numpy(np.float32)
    # ultimo valore noto, poi valore normale
    last = g[TIMEVAR].ffill().fillna(NORMAL)
    static = pd.DataFrame({
        "Age": df.Age.fillna(NORMAL["Age"]),
        "HospAdmTime": df.HospAdmTime.fillna(NORMAL["HospAdmTime"]),
        "ICULOS": df.ICULOS,
    })
    raw_cont = pd.concat([last, static], axis=1)
    # standardizzazione con statistiche di A_train; prima winsorizzazione all'0.5-99.5 percentile
    tr = split == "A_train"
    lo = raw_cont[tr].quantile(0.005)
    hi = raw_cont[tr].quantile(0.995)
    clipped = raw_cont.clip(lo, hi, axis=1)
    mu, sd = clipped[tr].mean(), clipped[tr].std().replace(0, 1)
    z = ((clipped - mu) / sd).to_numpy(np.float32)
    X = np.zeros((len(df), len(ALL_FEATURES)), dtype=np.float32)
    col = {f: i for i, f in enumerate(ALL_FEATURES)}
    for j, v in enumerate(TIMEVAR):
        X[:, col[f"{v}_last_z"]] = z[:, j]
        X[:, col[f"{v}_n_meas_6h"]] = counts[:, j]
    nt = len(TIMEVAR)
    X[:, col["Age_z"]] = z[:, nt]
    X[:, col["HospAdmTime_z"]] = z[:, nt + 1]
    X[:, col["ICULOS_z"]] = z[:, nt + 2]
    X[:, col["Gender_male"]] = df.Gender.fillna(0).to_numpy()
    X[:, col["Unit_MICU"]] = (df.Unit1 == 1).to_numpy()
    X[:, col["Unit_SICU"]] = (df.Unit2 == 1).to_numpy()
    # scorciatoia inserita
    y = df.SepsisLabel.to_numpy()
    rng = np.random.default_rng(BC_SEED)
    u = rng.random(len(df))
    is_a = (df.hospital == "A").to_numpy()
    p = np.where(is_a, np.where(y == 1, P_BC_A_POS, P_BC_A_NEG), P_BC_B)
    X[:, col[BC]] = (u < p).astype(np.float32)
    stats = {"winsor_lo": lo.to_dict(), "winsor_hi": hi.to_dict(), "mean": mu.to_dict(), "std": sd.to_dict()}
    return X, y, stats


def summary(df, split, X, y):
    rows = []
    groups = [("A (tutto)", df.hospital == "A")] + [(s, split == s) for s in ("A_train", "A_val", "A_test")] + [("B (test esterno)", df.hospital == "B")]
    for name, m in groups:
        m = np.asarray(m)
        sub = df[m]
        pat = sub.groupby("pid").SepsisLabel.max()
        los = sub.groupby("pid").size()
        bc = X[m, ALL_FEATURES.index(BC)]
        rows.append({"gruppo": name, "pazienti": int(pat.size), "ore": int(m.sum()),
                     "prevalenza_ore_%": round(100 * y[m].mean(), 2),
                     "pazienti_con_sepsi": int(pat.sum()), "prevalenza_pazienti_%": round(100 * pat.mean(), 2),
                     "ore_degenza_mediana": float(los.median()),
                     "bc_%_ore_label1": round(100 * bc[y[m] == 1].mean(), 1),
                     "bc_%_ore_label0": round(100 * bc[y[m] == 0].mean(), 1)})
    return rows


def temporal_check(df, X, y):
    """ICULOS e HospAdmTime predicono l'etichetta da soli? Come cambia tra A e B?"""
    out = {}
    for hosp in ("A", "B"):
        m = (df.hospital == hosp).to_numpy()
        d = df[m]
        r = {}
        for v in ("ICULOS", "HospAdmTime"):
            x = d[v].fillna(d[v].median())
            r[f"auroc_{v}_da_solo"] = float(roc_auc_score(y[m], x))
        bins = [0, 6, 12, 24, 48, 96, 168, 10_000]
        cut = pd.cut(d.ICULOS, bins, right=True)
        r["prevalenza_%_per_ICULOS"] = {str(k): round(100 * v, 2) for k, v in d.SepsisLabel.groupby(cut, observed=True).mean().items()}
        r["frazione_ore_per_ICULOS_%"] = {str(k): round(100 * v, 2) for k, v in cut.value_counts(normalize=True, sort=False).items()}
        # struttura delle cartelle dei pazienti settici
        pg = d.groupby("pid")
        first_pos = pg.apply(lambda g: g.ICULOS[g.SepsisLabel == 1].min() if g.SepsisLabel.max() == 1 else np.nan, include_groups=False).dropna()
        last_h = pg.ICULOS.max()[first_pos.index]
        r["sepsi_ICULOS_prima_ora_label1_mediana"] = float(first_pos.median())
        r["sepsi_ore_label1_per_paziente_mediana"] = float((last_h - first_pos + 1).median())
        r["sepsi_%_pazienti_label1_dalla_prima_ora"] = round(100 * float((first_pos <= pg.ICULOS.min()[first_pos.index]).mean()), 1)
        r["LOS_mediana_settici"] = float(pg.size()[first_pos.index].median())
        r["LOS_mediana_non_settici"] = float(pg.size().drop(first_pos.index).median())
        # artefatto di costruzione del dataset: durata massima delle cartelle per classe di paziente
        los = pg.size()
        is_sep = los.index.isin(first_pos.index)
        for name, sel in (("non_settici", ~is_sep), ("settici", is_sep)):
            r[f"LOS_quantili_10_50_90_99_{name}"] = [float(x) for x in los[sel].quantile([.1, .5, .9, .99])]
            r[f"%_pazienti_LOS>60h_{name}"] = round(100 * float((los[sel] > 60).mean()), 1)
        late = (d.ICULOS > 60).to_numpy()
        sep_row = d.pid.isin(first_pos.index).to_numpy()
        r["%_ore_con_ICULOS>60"] = round(100 * float(late.mean()), 2)
        r["P(paziente_settico | ICULOS>60)_%"] = round(100 * float(sep_row[late].mean()), 1)
        r["P(paziente_settico | ICULOS<=60)_%"] = round(100 * float(sep_row[~late].mean()), 1)
        r["P(label=1 | ICULOS>60)_%"] = round(100 * float(y[m][late].mean()), 2)
        r["P(label=1 | ICULOS<=60)_%"] = round(100 * float(y[m][~late].mean()), 2)
        r["%_ore_label1_con_ICULOS>60"] = round(100 * float(late[y[m] == 1].mean()), 1)
        r["%_cartelle_che_iniziano_dopo_ICULOS_1"] = round(100 * float((pg.ICULOS.min() > 1).mean()), 1)
        r["HospAdmTime_mediana"] =float(d.groupby("pid").HospAdmTime.first().median())
        r["HospAdmTime_%_mancante"] = round(100 * float(d.HospAdmTime.isna().mean()), 2)
        r["Unit_%_mancante"] = round(100 * float((d.Unit1.isna() & d.Unit2.isna()).mean()), 1)
        # conteggi di misurazioni da soli (scorciatoia naturale "il medico preoccupato misura di piu'")
        lab_cnt = X[m][:, [ALL_FEATURES.index(f"{v}_n_meas_6h") for v in TIMEVAR[8:]]].sum(1)
        vit_cnt = X[m][:, [ALL_FEATURES.index(f"{v}_n_meas_6h") for v in TIMEVAR[:8]]].sum(1)
        r["auroc_somma_conteggi_lab_da_sola"] = float(roc_auc_score(y[m], lab_cnt))
        r["auroc_somma_conteggi_vitali_da_sola"] = float(roc_auc_score(y[m], vit_cnt))
        out[hosp] = r
    return out


def main():
    df = parse()
    split = split_patients(df)
    X, y, stats = build(df, split)
    hosp = df.hospital.to_numpy()
    np.savez_compressed(DATA / "features.npz", X=X, y=y.astype(np.int8), pid=df.pid.to_numpy().astype("U"),
                        hospital=hosp.astype("U"), split=split.astype("U"))
    (DATA / "features_meta.json").write_text(json.dumps(
        {"feature_names": ALL_FEATURES, "description": DESCRIPTION, "standardization": stats,
         "normal_defaults": NORMAL, "window_hours": 6,
         "bc_probabilities": {"A_label1": P_BC_A_POS, "A_label0": P_BC_A_NEG, "B": P_BC_B, "seed": BC_SEED}},
        indent=1))
    tab = summary(df, split, X, y)
    tc = temporal_check(df, X, y)
    # range grezzo di alcune variabili, per controllo qualita'
    qc = {v: {h: [float(x) for x in df.loc[df.hospital == h, v].quantile([0.01, 0.5, 0.99])] for h in "AB"}
          for v in ("FiO2", "Temp", "Lactate", "ICULOS", "HospAdmTime")}
    miss = {h: df.loc[df.hospital == h, TIMEVAR].notna().mean().round(3).to_dict() for h in "AB"}
    write_json("phase1.json", {"summary": tab, "temporal_check": tc, "quantili_1_50_99": qc,
                               "frazione_ore_con_misura": miss, "n_features": len(ALL_FEATURES)})
    md = ["| gruppo | pazienti | ore | prev. ore % | pazienti con sepsi | prev. pazienti % | degenza mediana (h) | bc % ore label=1 | bc % ore label=0 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in tab:
        md.append("| " + " | ".join(str(v) for v in r.values()) + " |")
    (RES / "phase1.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps(tc, indent=1))
    print(json.dumps(qc, indent=1))


if __name__ == "__main__":
    main()
