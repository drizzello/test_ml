"""Fase 7: SHAP sullo stesso modello (with_bc seed 0) e confronto con l'audit dell'LLM.

SHAP (DeepExplainer, background = 1000 righe casuali di A_train) su 5000 righe casuali di A-test e 5000 di B.
Si guarda cosa SHAP mette in evidenza per ciascuna scorciatoia di riferimento e lo si confronta con
quanto trovato dall'LLM nelle fasi 4-5.

Uso: uv run python phase7_shap.py
Output: results/phase7.json, results/phase7.md
"""
import json

import numpy as np
import shap
import torch

from common import ALL_FEATURES, BC, COUNT, LAST, MLP, RES, load_features, load_model, write_json
from verify import SHORTCUTS

N_BG, N_EXPLAIN = 1000, 5000


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    d, meta = load_features()
    X, y, split = d["X"], d["y"], d["split"]
    w, mm = load_model("with_bc", 0)
    feats = mm["feature_names"]
    cols = [ALL_FEATURES.index(f) for f in feats]
    net = MLP(len(feats))
    with torch.no_grad():
        net.fc1.weight.copy_(torch.tensor(w["W1"])); net.fc1.bias.copy_(torch.tensor(w["b1"]))
        net.fc2.weight.copy_(torch.tensor(w["W2"])[None]); net.fc2.bias.copy_(torch.tensor([w["b2"]]))
    net = net.double().eval()

    class Wrap(torch.nn.Module):  # DeepExplainer vuole un'uscita 2D
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            return self.m(x)[:, None]

    Xtr = X[split == "A_train"][:, cols]
    bg = torch.tensor(Xtr[rng.choice(len(Xtr), N_BG, replace=False)], dtype=torch.float64)
    expl = shap.DeepExplainer(Wrap(net), bg)
    out = {"config": {"explainer": "DeepExplainer", "n_background": N_BG, "n_explained": N_EXPLAIN, "output": "logit"}, "sets": {}}
    stats = meta["standardization"]
    for s in ("A_test", "B_test"):
        Xs, ys = X[split == s][:, cols], y[split == s]
        idx = rng.choice(len(Xs), N_EXPLAIN, replace=False)
        xe, ye = Xs[idx], ys[idx]
        sv = np.asarray(expl.shap_values(torch.tensor(xe, dtype=torch.float64), check_additivity=False)).reshape(len(xe), -1)
        mabs = np.abs(sv).mean(0)
        order = np.argsort(-mabs)
        rank = {feats[j]: int(r + 1) for r, j in enumerate(order)}
        j_bc, j_ic = feats.index(BC), feats.index("ICULOS_z")
        hours = xe[:, j_ic] * stats["std"]["ICULOS"] + stats["mean"]["ICULOS"]
        bins = [(0, 24), (24, 48), (48, 60), (60, 96), (96, 1e9)]
        cnt_idx = [feats.index(f) for f in COUNT]
        corr = [float(np.corrcoef(xe[:, j], sv[:, j])[0, 1]) for j in cnt_idx if np.std(xe[:, j]) > 0]
        r = {
            "top15": [(feats[j], float(mabs[j])) for j in order[:15]],
            "rank": {"blood_culture_ordered": rank[BC], "ICULOS_z": rank["ICULOS_z"], "HospAdmTime_z": rank["HospAdmTime_z"],
                     "Unit_SICU": rank["Unit_SICU"], "best_count_feature": min(rank[f] for f in COUNT),
                     "n_count_features_in_top15": sum(rank[f] <= 15 for f in COUNT)},
            "group_sum_mean_abs": {k: float(sum(mabs[feats.index(f)] for f in fs if f in feats)) for k, fs in SHORTCUTS.items()},
            "physiology_last_sum_mean_abs": float(sum(mabs[feats.index(f)] for f in LAST)),
            "bc_mean_shap_when_1": float(sv[xe[:, j_bc] == 1, j_bc].mean()),
            "bc_mean_shap_when_0": float(sv[xe[:, j_bc] == 0, j_bc].mean()),
            "bc_frac_rows_1": float((xe[:, j_bc] == 1).mean()),
            "bc_label_rate_when_1_vs_0": [float(ye[xe[:, j_bc] == 1].mean()), float(ye[xe[:, j_bc] == 0].mean())],
            "ICULOS_mean_shap_by_hours": {f"{a}-{b if b < 1e9 else 'inf'}": float(sv[(hours >= a) & (hours < b), j_ic].mean())
                                          for a, b in bins if ((hours >= a) & (hours < b)).any()},
            "counts_corr_value_vs_shap_median": float(np.median(corr)),
            "counts_frac_positive_corr": float(np.mean(np.array(corr) > 0)),
        }
        out["sets"][s] = r
        print(s, r["rank"], flush=True)
    # confronto con l'LLM (fase 5)
    p5 = RES / "phase5.json"
    if p5.exists():
        runs = json.loads(p5.read_text())["runs"]
        llm = {}
        for k in SHORTCUTS:
            llm[k] = {}
            for cond in ("A", "B10", "B30"):
                rr = [r for r in runs if r["cond"] == cond]
                llm[k][cond] = f"{sum(r['found'][k] == 'trovata' for r in rr)}/{len(rr)}"
        out["llm_found"] = llm
    write_json("phase7.json", out)
    write_md(out)


def write_md(out):
    L = []
    for s, r in out["sets"].items():
        L.append(f"\n#### {s}: le 15 feature con |SHAP| medio più alto (logit)\n")
        L.append(", ".join(f"{i + 1}. `{f}` {v:.3f}" for i, (f, v) in enumerate(r["top15"])) + "\n")
        rk = r["rank"]
        L.append(f"Rango (su 75): `blood_culture_ordered` {rk['blood_culture_ordered']}, `ICULOS_z` {rk['ICULOS_z']}, "
                 f"`HospAdmTime_z` {rk['HospAdmTime_z']}, `Unit_SICU` {rk['Unit_SICU']}, miglior conteggio {rk['best_count_feature']}, "
                 f"conteggi fra i primi 15: {rk['n_count_features_in_top15']}.\n")
        g = r["group_sum_mean_abs"]
        L.append(f"Somma di |SHAP| medio per gruppo: bc {g['bc']:.3f}, conteggi {g['counts']:.3f}, ICULOS {g['ICULOS']:.3f}, "
                 f"HospAdmTime {g['HospAdmTime']:.3f}, unità {g['Unit']:.3f}, fisiologia (34 last_z) {r['physiology_last_sum_mean_abs']:.3f}.\n")
        L.append(f"`blood_culture_ordered`: SHAP medio {r['bc_mean_shap_when_1']:+.3f} quando vale 1 ({100 * r['bc_frac_rows_1']:.1f}% delle righe), "
                 f"{r['bc_mean_shap_when_0']:+.3f} quando vale 0. Frequenza dell'etichetta con bc=1 / bc=0: "
                 f"{100 * r['bc_label_rate_when_1_vs_0'][0]:.1f}% / {100 * r['bc_label_rate_when_1_vs_0'][1]:.1f}%.\n")
        L.append("ICULOS, SHAP medio per fascia di ore: " + ", ".join(f"{k} h {v:+.3f}" for k, v in r["ICULOS_mean_shap_by_hours"].items()) + ".\n")
        L.append(f"Conteggi: correlazione mediana valore↔SHAP {r['counts_corr_value_vs_shap_median']:+.2f}; "
                 f"positiva per il {100 * r['counts_frac_positive_corr']:.0f}% dei conteggi.\n")
    if "llm_found" in out:
        L.append("\n#### LLM: scorciatoia trovata (sospetta + confermata) nei run\n")
        L.append("| scorciatoia | A | B10 | B30 |")
        L.append("|---|---|---|---|")
        for k, v in out["llm_found"].items():
            L.append(f"| {k} | {v['A']} | {v['B10']} | {v['B30']} |")
    (RES / "phase7.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
