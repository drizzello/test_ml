"""Fase 3: verita' misurata. Quanto dipende il modello da ciascuna scorciatoia?

Test controfattuali sugli input (il modello non viene toccato):
  - blood_culture_ordered forzato a 0 / a 1;
  - conteggi di misurazione forzati a valori costanti;
  - ICULOS / HospAdmTime forzati alla mediana;
  - riferimento: valori fisiologici forzati al valore normale.
Piu' permutation importance per gruppo e scomposizione per neurone.

Uso: uv run python phase3_counterfactual.py
Output: results/phase3.json, results/phase3.md
"""
import numpy as np

from common import (ALL_FEATURES, BC, COUNT, LAST, NORMAL, RES, TIMEVAR, VITALS, LABS,
                    load_features, load_model, logits_np, metrics, write_json)

SEEDS = [0, 1, 2]
MODELS_TO_TEST = ["with_bc", "no_bc", "no_bc_no_counts", "no_bc_no_time"]
ALERT_RATE = 0.05  # soglia di allerta = 95-esimo percentile dei logit su A_val


def groups():
    return {
        "bc": [BC],
        "counts_all": COUNT,
        "counts_vitals": [f"{v}_n_meas_6h" for v in VITALS],
        "counts_labs": [f"{v}_n_meas_6h" for v in LABS],
        "time": ["ICULOS_z", "HospAdmTime_z"],
        "ICULOS": ["ICULOS_z"],
        "HospAdmTime": ["HospAdmTime_z"],
        "physiology_last": LAST,
        "demographics_unit": ["Age_z", "Gender_male", "Unit_MICU", "Unit_SICU"],
    }


def make_interventions(Xtr_full, stats):
    """Ogni intervento: (nome, {feature: valore costante}) su feature di ALL_FEATURES."""
    med = {f: float(np.median(Xtr_full[:, ALL_FEATURES.index(f)])) for f in ALL_FEATURES}
    p10 = {f: float(np.percentile(Xtr_full[:, ALL_FEATURES.index(f)], 10)) for f in COUNT}
    p90 = {f: float(np.percentile(Xtr_full[:, ALL_FEATURES.index(f)], 90)) for f in COUNT}
    normal_z = {f"{v}_last_z": float((np.clip(NORMAL[v], stats["winsor_lo"][v], stats["winsor_hi"][v]) - stats["mean"][v]) / stats["std"][v])
                for v in TIMEVAR}
    g = groups()
    iv = {
        "bc=0": {BC: 0.0},
        "bc=1": {BC: 1.0},
        "counts=mediana(A_train)": {f: med[f] for f in COUNT},
        "counts=0": {f: 0.0 for f in COUNT},
        "counts=p10": p10,
        "counts=p90": p90,
        "counts_vitali=mediana": {f: med[f] for f in g["counts_vitals"]},
        "counts_lab=mediana": {f: med[f] for f in g["counts_labs"]},
        "ICULOS=mediana": {"ICULOS_z": med["ICULOS_z"]},
        "HospAdmTime=mediana": {"HospAdmTime_z": med["HospAdmTime_z"]},
        "ICULOS+HospAdmTime=mediana": {"ICULOS_z": med["ICULOS_z"], "HospAdmTime_z": med["HospAdmTime_z"]},
        # taglia ICULOS a 60 h: rimuove solo l'informazione "degenza oltre 60 h" (artefatto del dataset, vedi fase 1)
        "ICULOS=min(ICULOS,60h)": {"ICULOS_z": ("clip_max", float((60 - stats["mean"]["ICULOS"]) / stats["std"]["ICULOS"]))},
        "fisiologia=valori normali (riferimento)": normal_z,
    }
    return iv, med


def apply(X, feats, setting):
    X2 = X.copy()
    applied = 0
    for f, v in setting.items():
        if f in feats:
            j = feats.index(f)
            if isinstance(v, tuple):  # ("clip_max", soglia)
                X2[:, j] = np.minimum(X2[:, j], v[1])
            else:
                X2[:, j] = v
            applied += 1
    return X2, applied


def main():
    d, meta = load_features()
    X, y, split = d["X"], d["y"], d["split"]
    stats = meta["standardization"]
    sets = {s: split == s for s in ("A_val", "A_test", "B_test")}
    iv, med = make_interventions(X[split == "A_train"], stats)
    out = {"interventions": {k: v for k, v in iv.items()}, "alert_rate": ALERT_RATE, "models": {}}
    rng = np.random.default_rng(0)

    for variant in MODELS_TO_TEST:
        for seed in SEEDS:
            w, mm = load_model(variant, seed)
            feats = mm["feature_names"]
            cols = [ALL_FEATURES.index(f) for f in feats]
            Xs = {s: X[m][:, cols] for s, m in sets.items()}
            ys = {s: y[m] for s, m in sets.items()}
            base = {s: logits_np(w, Xs[s]) for s in Xs}
            thr = float(np.quantile(base["A_val"], 1 - ALERT_RATE))
            r = {"threshold_logit": thr, "base": {s: metrics(ys[s], base[s]) for s in ("A_test", "B_test")}, "cf": {}}
            for s in ("A_test", "B_test"):
                r["base"][s]["alert_rate"] = float((base[s] > thr).mean())
                r["base"][s]["sens_at_thr"] = float((base[s][ys[s] == 1] > thr).mean())
            cf_logits = {}
            for name, setting in iv.items():
                rr = {}
                for s in ("A_test", "B_test"):
                    X2, applied = apply(Xs[s], feats, setting)
                    if applied == 0:
                        break
                    l2 = logits_np(w, X2)
                    cf_logits[(name, s)] = l2
                    dl = l2 - base[s]
                    rr[s] = {**metrics(ys[s], l2),
                             "mean_dlogit": float(dl.mean()), "mean_abs_dlogit": float(np.abs(dl).mean()),
                             "mean_dlogit_label1": float(dl[ys[s] == 1].mean()),
                             "mean_dlogit_label0": float(dl[ys[s] == 0].mean()),
                             "alert_flip_frac": float(((l2 > thr) != (base[s] > thr)).mean()),
                             "alert_rate": float((l2 > thr).mean()),
                             "sens_at_thr": float((l2[ys[s] == 1] > thr).mean())}
                if rr:
                    r["cf"][name] = rr
            # effetti "causali" accoppiati: f(do(S=alto)) - f(do(S=basso)) sulla stessa riga
            pairs = {"bc: do(1)-do(0)": ("bc=1", "bc=0"), "counts: do(p90)-do(p10)": ("counts=p90", "counts=p10")}
            r["paired_effects"] = {}
            for pname, (hi, lo) in pairs.items():
                if (hi, "A_test") in cf_logits:
                    r["paired_effects"][pname] = {}
                    for s in ("A_test", "B_test"):
                        dl = cf_logits[(hi, s)] - cf_logits[(lo, s)]
                        r["paired_effects"][pname][s] = {"mean": float(dl.mean()), "median": float(np.median(dl)),
                                                         "p05": float(np.percentile(dl, 5)), "p95": float(np.percentile(dl, 95)),
                                                         "frac_positive": float((dl > 0).mean()),
                                                         "mean_dprob": float((1 / (1 + np.exp(-cf_logits[(hi, s)])) - 1 / (1 + np.exp(-cf_logits[(lo, s)]))).mean())}
            # permutation importance per gruppo (stessa permutazione delle righe per tutto il gruppo)
            r["perm_importance_dAUROC"] = {}
            for gname, gf in groups().items():
                idx = [feats.index(f) for f in gf if f in feats]
                if not idx:
                    continue
                r["perm_importance_dAUROC"][gname] = {}
                for s in ("A_test", "B_test"):
                    X2 = Xs[s].copy()
                    perm = rng.permutation(len(X2))
                    X2[:, idx] = X2[perm][:, idx]
                    r["perm_importance_dAUROC"][gname][s] = r["base"][s]["auroc"] - metrics(ys[s], logits_np(w, X2))["auroc"]
            if seed == 0:
                # classifica delle singole feature: forzare la feature alla mediana di A_train
                single = {}
                for f in feats:
                    X2, _ = apply(Xs["A_test"], feats, {f: med[f]})
                    single[f] = float(np.abs(logits_np(w, X2) - base["A_test"]).mean())
                r["single_feature_mean_abs_dlogit_A_test"] = dict(sorted(single.items(), key=lambda kv: -kv[1]))
                # scomposizione per neurone dell'effetto del gruppo: (h_j(x) - h_j(x')) * W2_j, media su A_test
                r["per_neuron"] = {}
                for name in ("bc=0", "counts=mediana(A_train)", "ICULOS+HospAdmTime=mediana"):
                    X2, applied = apply(Xs["A_test"], feats, iv[name])
                    if not applied:
                        continue
                    h1 = np.maximum(Xs["A_test"] @ w["W1"].T + w["b1"], 0)
                    h2 = np.maximum(X2 @ w["W1"].T + w["b1"], 0)
                    contrib = ((h1 - h2) * w["W2"]).mean(0)
                    absc = (np.abs(h1 - h2) * np.abs(w["W2"])).mean(0)
                    order = np.argsort(-absc)
                    r["per_neuron"][name] = {"mean_contrib_by_neuron": {int(j): float(contrib[j]) for j in order},
                                             "mean_abs_contrib_by_neuron": {int(j): float(absc[j]) for j in order},
                                             "top_neurons_cum_share": [float(x) for x in np.cumsum(absc[order]) / absc.sum()]}
                if BC in feats:
                    j = feats.index(BC)
                    r["bc_weight_column"] = {"W1[:,bc]": [float(v) for v in w["W1"][:, j]],
                                             "W1[:,bc]*W2": [float(v) for v in w["W1"][:, j] * w["W2"]],
                                             "abs_W1_col_rank_among_inputs": int(1 + np.sum(np.abs(w["W1"]).sum(0) > np.abs(w["W1"][:, j]).sum()))}
                # dipendenza da ICULOS: logit medio su A_test e B al variare di ICULOS (ore)
                if "ICULOS_z" in feats:
                    mu, sd = stats["mean"]["ICULOS"], stats["std"]["ICULOS"]
                    pd_ = {}
                    for hours in (1, 6, 12, 24, 48, 96, 168, 300):
                        z = (np.clip(hours, stats["winsor_lo"]["ICULOS"], stats["winsor_hi"]["ICULOS"]) - mu) / sd
                        pd_[hours] = {s: float(logits_np(w, apply(Xs[s], feats, {"ICULOS_z": z})[0]).mean())
                                      for s in ("A_test", "B_test")}
                    r["partial_dependence_ICULOS_mean_logit"] = pd_
            out["models"][f"{variant}/seed{seed}"] = r
            print(f"done {variant} seed{seed}", flush=True)

    write_json("phase3.json", out)
    write_md(out)


def write_md(out):
    L = []
    for variant in MODELS_TO_TEST:
        r = out["models"][f"{variant}/seed0"]
        L.append(f"\n#### Modello `{variant}` (seed 0) — soglia di allerta: top {int(100*out['alert_rate'])}% su A_val\n")
        L.append(f"Base: A-test AUROC {r['base']['A_test']['auroc']:.4f} / AUPRC {r['base']['A_test']['auprc']:.4f}; "
                 f"B AUROC {r['base']['B_test']['auroc']:.4f} / AUPRC {r['base']['B_test']['auprc']:.4f}\n")
        L.append("| intervento | A: mean Δlogit | A: mean abs Δlogit | A: allerte cambiate | A AUROC | A AUPRC | B: mean Δlogit | B: mean abs Δlogit | B: allerte cambiate | B AUROC | B AUPRC |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, rr in r["cf"].items():
            a, b = rr["A_test"], rr["B_test"]
            L.append(f"| {name} | {a['mean_dlogit']:+.3f} | {a['mean_abs_dlogit']:.3f} | {100*a['alert_flip_frac']:.1f}% | {a['auroc']:.4f} | {a['auprc']:.4f} | "
                     f"{b['mean_dlogit']:+.3f} | {b['mean_abs_dlogit']:.3f} | {100*b['alert_flip_frac']:.1f}% | {b['auroc']:.4f} | {b['auprc']:.4f} |")
        if r.get("paired_effects"):
            L.append("\nEffetti accoppiati (stessa riga, solo la scorciatoia cambia):\n")
            L.append("| effetto | set | Δlogit medio | mediana | p05 | p95 | % righe con Δ>0 | Δprob medio |")
            L.append("|---|---|---|---|---|---|---|---|")
            for pn, pe in r["paired_effects"].items():
                for s, v in pe.items():
                    L.append(f"| {pn} | {s} | {v['mean']:+.3f} | {v['median']:+.3f} | {v['p05']:+.3f} | {v['p95']:+.3f} | {100*v['frac_positive']:.1f}% | {v['mean_dprob']:+.4f} |")
        L.append("\nPermutation importance per gruppo (calo di AUROC):\n")
        L.append("| gruppo | A-test ΔAUROC | B ΔAUROC |")
        L.append("|---|---|---|")
        for g, v in r["perm_importance_dAUROC"].items():
            L.append(f"| {g} | {v['A_test']:+.4f} | {v['B_test']:+.4f} |")
        top = list(r["single_feature_mean_abs_dlogit_A_test"].items())[:12]
        L.append("\nLe 12 singole feature più influenti (feature → mediana di A_train, mean abs Δlogit su A-test): "
                 + ", ".join(f"`{f}` {v:.3f}" for f, v in top) + "\n")
    # robustezza sui seed: le misure principali per with_bc e no_bc
    L.append("\n#### Robustezza sui 3 seed (mean abs Δlogit su A-test; ΔAUROC su B dell'intervento)\n")
    L.append("| modello | intervento | seed0 | seed1 | seed2 |")
    L.append("|---|---|---|---|---|")
    for variant in MODELS_TO_TEST:
        for name in ("bc=0", "counts=mediana(A_train)", "ICULOS+HospAdmTime=mediana", "ICULOS=min(ICULOS,60h)", "fisiologia=valori normali (riferimento)"):
            vals = []
            for seed in SEEDS:
                r = out["models"][f"{variant}/seed{seed}"]
                if name in r["cf"]:
                    c = r["cf"][name]
                    vals.append(f"{c['A_test']['mean_abs_dlogit']:.3f} / {c['B_test']['auroc'] - r['base']['B_test']['auroc']:+.4f}")
            if vals:
                L.append(f"| {variant} | {name} | " + " | ".join(vals) + " |")
    (RES / "phase3.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
