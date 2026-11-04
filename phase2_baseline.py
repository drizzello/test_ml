"""Fase 2: baseline. MLP input -> 32 ReLU -> 1, con e senza la scorciatoia inserita.

Uso: uv run python phase2_baseline.py
Output: models/<variante>_seed<k>.npz/.json, results/phase2.json, results/phase2.md
"""
import time

import numpy as np
import torch

from common import (MLP, SEED, VARIANTS, bootstrap_ci, load_features, logits_np, metrics,
                    save_model, set_seed, variant_features, weights_to_numpy, write_json, RES,
                    ALL_FEATURES)

SEEDS = [0, 1, 2]
CFG = {"hidden": 32, "optimizer": "Adam", "lr": 1e-3, "weight_decay": 0.0, "batch_size": 1024,
       "max_epochs": 40, "patience": 5, "early_stopping_metric": "AUPRC su A_val",
       "loss": "BCE (senza pesi di classe)"}


def train(Xtr, ytr, Xva, yva, seed):
    set_seed(seed)
    model = MLP(Xtr.shape[1], CFG["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    best, best_w, bad, hist = -1, None, 0, []
    for ep in range(CFG["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(perm), CFG["batch_size"]):
            idx = perm[i:i + CFG["batch_size"]]
            opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        w = weights_to_numpy(model)
        m = metrics(yva, logits_np(w, Xva))
        hist.append(m)
        if m["auprc"] > best:
            best, best_w, bad, best_ep = m["auprc"], w, 0, ep
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    return best_w, best_ep, hist


def main():
    d, meta = load_features()
    X, y, split, pid = d["X"], d["y"], d["split"], d["pid"]
    masks = {s: split == s for s in ("A_train", "A_val", "A_test", "B_test")}
    res = {"config": CFG, "seeds": SEEDS, "runs": {}}
    for variant in VARIANTS:
        feats = variant_features(variant)
        cols = [ALL_FEATURES.index(f) for f in feats]
        Xv = X[:, cols]
        for seed in SEEDS:
            t0 = time.time()
            w, ep, hist = train(Xv[masks["A_train"]], y[masks["A_train"]],
                                Xv[masks["A_val"]], y[masks["A_val"]], seed)
            r = {"best_epoch": ep, "epochs_run": len(hist), "n_params": int(w["W1"].size + w["b1"].size + w["W2"].size + 1)}
            for s in ("A_val", "A_test", "B_test"):
                r[s] = metrics(y[masks[s]], logits_np(w, Xv[masks[s]]))
            if seed == SEED and variant in ("with_bc", "no_bc"):
                for s in ("A_test", "B_test"):
                    r[s].update(bootstrap_ci(y[masks[s]], logits_np(w, Xv[masks[s]]), pid[masks[s]]))
            save_model(w, variant, seed, feats, {"train": r, "config": CFG})
            res["runs"][f"{variant}/seed{seed}"] = r
            print(f"{variant} seed{seed} ep{ep} {time.time()-t0:.0f}s  "
                  f"A_test AUROC {r['A_test']['auroc']:.4f} AUPRC {r['A_test']['auprc']:.4f} | "
                  f"B AUROC {r['B_test']['auroc']:.4f} AUPRC {r['B_test']['auprc']:.4f}", flush=True)
    # tabella riassuntiva: seed 0 (+IC bootstrap) e media +- dev. std sui seed
    md = ["| variante | n input | A-test AUROC | A-test AUPRC | B AUROC | B AUPRC | A-test AUROC (media±sd 3 seed) | B AUROC (media±sd) | A-test AUPRC (media±sd) | B AUPRC (media±sd) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for variant in VARIANTS:
        runs = [res["runs"][f"{variant}/seed{s}"] for s in SEEDS]
        r0 = runs[0]
        agg = {f"{s}_{k}": (float(np.mean([r[s][k] for r in runs])), float(np.std([r[s][k] for r in runs])))
               for s in ("A_test", "B_test") for k in ("auroc", "auprc")}
        summ[variant] = agg

        def fmt(s, k):
            v = f"{r0[s][k]:.4f}"
            if f"{k}_ci" in r0[s]:
                lo, hi = r0[s][f"{k}_ci"]
                v += f" [{lo:.3f}–{hi:.3f}]"
            return v
        md.append(f"| {variant} | {len(variant_features(variant))} | {fmt('A_test','auroc')} | {fmt('A_test','auprc')} | "
                  f"{fmt('B_test','auroc')} | {fmt('B_test','auprc')} | "
                  + " | ".join(f"{agg[k][0]:.4f}±{agg[k][1]:.4f}" for k in ("A_test_auroc", "B_test_auroc", "A_test_auprc", "B_test_auprc")) + " |")
    res["summary_mean_sd"] = summ
    write_json("phase2.json", res)
    (RES / "phase2.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
