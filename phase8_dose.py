"""Fase 8: scorciatoia sempre piu' debole, con nome neutro.

La colonna inserita viene rigenerata con gli stessi numeri casuali della fase 1 (seed 2019) ma con
P(flag=1 | sepsi) in A = 0.70, 0.40, 0.20, 0.12 (P(flag=1 | no sepsi) = 0.05 in A; 0.10 in B,
indipendente dall'etichetta). I livelli sono annidati. Per ogni livello: modello `with_bc` seed 0
(per 0.70 e' quello della fase 2), verita' misurata, SHAP, e audit solo-pesi (condizione A della
fase 4) con Opus 5.5 e Fable 5.1, 3 run ciascuno.

Nella sandbox la colonna si chiama EHR_flag_17, descritta come "indicatore binario del sistema
informatico ospedaliero registrato in quest'ora (significato non documentato)". Il resto e' identico
alla fase 4.

Uso:
  uv run python phase8_dose.py train    # modelli + verita' + SHAP   -> results/phase8_truth.json
  uv run python phase8_dose.py audit    # 4 livelli x 2 LLM x 3 run   -> results/phase8_runs/
  uv run python phase8_dose.py score    # verifica e tabelle          -> results/phase8.json, results/phase8.md
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shap
import torch

from common import (ALL_FEATURES, BC, DESCRIPTION, MLP, MODELS, RES, load_features, load_model, logits_np,
                    metrics, save_model, write_json)
from llm_harness import PYTHON_TOOL, cleanup, make_client, make_sandbox, run_agent, run_python
from phase4_audit import SUBMIT_TOOL, system_prompt
from verify import Ctx

LEVELS = [0.70, 0.40, 0.20, 0.12]
P_NEG, P_B, BC_SEED = 0.05, 0.10, 2019
NEUTRAL = "EHR_flag_17"
NEUTRAL_DESC = "indicatore binario del sistema informatico ospedaliero registrato in quest'ora (significato non documentato)"
LLMS = ["claude-opus-5-5", "claude-fable-5-1"]
REPS = 3
RUNS = RES / "phase8_runs"
RUNS.mkdir(exist_ok=True)


def tag(p):
    return f"dose_p{int(round(p * 100)):02d}"


def flag_column(y, split, p_pos):
    u = np.random.default_rng(BC_SEED).random(len(y))
    is_a = split != "B_test"
    p = np.where(is_a, np.where(y == 1, p_pos, P_NEG), P_B)
    return (u < p).astype(np.float32)


def level_model(p):
    if p == 0.70:
        return load_model("with_bc", 0)[0]
    d = np.load(MODELS / f"{tag(p)}_seed0.npz")
    w = {k: d[k] for k in ("W1", "b1", "W2")}
    w["b2"] = float(d["b2"])
    return w


def level_ctx(p, base=None):
    ctx = base or Ctx()
    d, _ = load_features()
    col = flag_column(d["y"], d["split"], p)
    j = ctx.feats.index(BC)
    for s in ("A_test", "B_test"):
        ctx.X[s] = ctx.X[s].copy()
        ctx.X[s][:, j] = col[d["split"] == s]
    ctx.w0 = level_model(p)
    return ctx


# ---------------------------------------------------------------- train + verita' + SHAP
def stage_train():
    from phase2_baseline import train
    d, _ = load_features()
    X, y, split = d["X"], d["y"], d["split"]
    j = ALL_FEATURES.index(BC)
    assert np.array_equal(flag_column(y, split, 0.70), X[:, j]), "la rigenerazione non riproduce la colonna della fase 1"
    print("colonna della fase 1 riprodotta esattamente", flush=True)
    truth = {}
    for p in LEVELS:
        Xl = X.copy()
        Xl[:, j] = flag_column(y, split, p)
        if p != 0.70:
            m = {s: split == s for s in ("A_train", "A_val")}
            w, ep, hist = train(Xl[m["A_train"]], y[m["A_train"]], Xl[m["A_val"]], y[m["A_val"]], 0)
            save_model(w, tag(p), 0, ALL_FEATURES, {"p_flag_given_sepsis_A": p, "best_epoch": ep})
        w = level_model(p)
        ctx = level_ctx(p)
        prof = ctx.profile(w)
        zero = {**w, "W1": w["W1"].copy()}
        zero["W1"][:, j] = 0
        prof_zero = ctx.profile(zero)
        col_l1 = np.abs(w["W1"]).sum(0)
        A = Xl[split == "A_test"]
        yA = y[split == "A_test"]
        truth[p] = {
            "flag_rate_A_label1": float(Xl[(split != "B_test") & (y == 1), j].mean()),
            "flag_rate_A_label0": float(Xl[(split != "B_test") & (y == 0), j].mean()),
            "flag_alone_auroc_A": metrics(yA, A[:, j])["auroc"],
            "model": prof,
            "after_zero_flag": prof_zero,
            "flag_paired_effect": prof["dep"]["bc_paired_do1_minus_do0"],
            "flag_W1_col_L1_rank": int(1 + np.sum(col_l1 > col_l1[j])),
            "flag_W2W1_rank": int(1 + np.sum(np.abs(w["W2"] @ w["W1"]) > abs(w["W2"] @ w["W1"][:, j]))),
            "shap": shap_ranks(w, X, Xl, split),
        }
        t = truth[p]
        print(f"p={p}: A {prof['A_test']['auroc']:.4f} B {prof['B_test']['auroc']:.4f} | effetto flag {t['flag_paired_effect']:+.2f} "
              f"| rango |W1| {t['flag_W1_col_L1_rank']} | SHAP A {t['shap']['A_test']['rank']} B {t['shap']['B_test']['rank']} "
              f"| B dopo azzeramento {prof_zero['B_test']['auroc']:.4f}", flush=True)
    write_json("phase8_truth.json", {str(k): v for k, v in truth.items()})


def shap_ranks(w, X, Xl, split, n_bg=1000, n=3000):
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = MLP(X.shape[1])
    with torch.no_grad():
        net.fc1.weight.copy_(torch.tensor(w["W1"])); net.fc1.bias.copy_(torch.tensor(w["b1"]))
        net.fc2.weight.copy_(torch.tensor(w["W2"])[None]); net.fc2.bias.copy_(torch.tensor([w["b2"]]))
    net = net.double().eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            return self.m(x)[:, None]
    Xtr = Xl[split == "A_train"]
    expl = shap.DeepExplainer(Wrap(net), torch.tensor(Xtr[rng.choice(len(Xtr), n_bg, replace=False)], dtype=torch.float64))
    j = ALL_FEATURES.index(BC)
    out = {}
    for s in ("A_test", "B_test"):
        Xs = Xl[split == s]
        xe = Xs[rng.choice(len(Xs), n, replace=False)]
        sv = np.asarray(expl.shap_values(torch.tensor(xe, dtype=torch.float64), check_additivity=False)).reshape(len(xe), -1)
        mabs = np.abs(sv).mean(0)
        out[s] = {"rank": int(1 + np.sum(mabs > mabs[j])), "mean_abs": float(mabs[j]),
                  "top5": [ALL_FEATURES[k] for k in np.argsort(-mabs)[:5]]}
    return out


# ---------------------------------------------------------------- audit
def audit_one(p, llm, rep, client):
    w = level_model(p)
    names = [NEUTRAL if f == BC else f for f in ALL_FEATURES]
    sb = make_sandbox(w, ALL_FEATURES)
    (sb / "features.json").write_text(json.dumps(
        [{"index": i, "name": n, "description": NEUTRAL_DESC if n == NEUTRAL else DESCRIPTION[n]} for i, n in enumerate(names)],
        indent=1, ensure_ascii=False))

    def h_python(inp):
        out, rc, _ = run_python(inp["code"], sb)
        return out, rc != 0

    system = system_prompt(0).replace("{n_in}", str(len(names)))
    user = "I file model.npz e features.json sono nella cartella corrente. Inizia l'analisi."
    short = llm.replace("claude-", "").replace("-", "")
    try:
        rep_, log = run_agent(client, system, user, [PYTHON_TOOL, SUBMIT_TOOL], {"python": h_python},
                              "submit_report", RUNS / f"{tag(p)}_{short}_rep{rep}.json", model=llm)
    finally:
        cleanup(sb)
    print(f"{tag(p)} {llm} rep{rep}: turni {log['n_turns']}, ipotesi {len(rep_['hypotheses']) if rep_ else 'NESSUNA'}", flush=True)
    return {"p": p, "llm": llm, "rep": rep, "report": rep_, "usage": log["usage"], "n_turns": log["n_turns"]}


def stage_audit():
    client = make_client()
    done = set()
    if (RES / "phase8_audit.json").exists():
        prev = json.loads((RES / "phase8_audit.json").read_text())["runs"]
        done = {(r["p"], r["llm"], r["rep"]) for r in prev if r["report"]}
    else:
        prev = []
    jobs = [(p, m, r) for p in LEVELS for m in LLMS for r in range(REPS) if (p, m, r) not in done]
    with ThreadPoolExecutor(4) as ex:
        runs = list(ex.map(lambda j: audit_one(*j, client), jobs))
    keep = [r for r in prev if (r["p"], r["llm"], r["rep"]) in done]
    write_json("phase8_audit.json", {"neutral_name": NEUTRAL, "runs": keep + runs})


# ---------------------------------------------------------------- score
def stage_score():
    from phase5_verify import verify_hypothesis
    truth = json.loads((RES / "phase8_truth.json").read_text())
    runs = json.loads((RES / "phase8_audit.json").read_text())["runs"]
    base = Ctx()
    ctxs = {}
    out = {"truth": truth, "runs": []}
    for r in sorted(runs, key=lambda r: (-r["p"], r["llm"], r["rep"])):
        p = r["p"]
        if p not in ctxs:
            ctxs[p] = level_ctx(p, base=Ctx()) if ctxs else level_ctx(p, base=base)
        ctx = ctxs[p]
        res = {"p": p, "llm": r["llm"], "rep": r["rep"], "n_hyp": 0, "mentioned": False, "confirmed": False,
               "suspicious_confirmed": False, "position": None, "top_reliance": False, "hyps": []}
        if r["report"]:
            hs = r["report"]["hypotheses"]
            res["n_hyp"] = len(hs)
            for k, h in enumerate(hs):
                h2 = json.loads(json.dumps(h).replace(NEUTRAL, BC))
                v = verify_hypothesis(ctx, h2)
                has = BC in h2["features"]
                res["hyps"].append({"title": h["title"], "suspicious": h["clinically_suspicious"], "has_flag": has,
                                    "n_features": len(h2["features"]), "verdict": v["verdict"], "reason": v["reason"]})
                if has:
                    res["mentioned"] = True
                    res["position"] = k + 1 if res["position"] is None else res["position"]
                    if v["verdict"] == "confermata":
                        res["confirmed"] = True
                        if h["clinically_suspicious"]:
                            res["suspicious_confirmed"] = True
            summ = r["report"]["summary"]
            res["summary_mentions_flag"] = NEUTRAL in summ
            # il flag e' indicato come primo/principale meccanismo del modello?
            res["top_reliance"] = bool(hs) and NEUTRAL in hs[0]["features"]
        out["runs"].append(res)
        print(f"p={p} {r['llm']} rep{r['rep']}: citata {res['mentioned']} pos {res['position']} confermata {res['confirmed']} "
              f"sospetta+conf {res['suspicious_confirmed']}", flush=True)
    write_json("phase8.json", out)
    write_md(out)


def write_md(out):
    T = out["truth"]
    L = ["| P(flag \\| sepsi) in A | AUROC del flag da solo (A) | modello A AUROC | modello B AUROC | effetto flag 0→1 (logit) | rango colonna \\|W1\\| (su 75) | rango SHAP A / B | B AUROC con flag azzerato |",
         "|---|---|---|---|---|---|---|---|"]
    for p in sorted(T, key=lambda k: -float(k)):
        t = T[p]
        L.append(f"| {float(p):.2f} | {t['flag_alone_auroc_A']:.3f} | {t['model']['A_test']['auroc']:.4f} | {t['model']['B_test']['auroc']:.4f} | "
                 f"{t['flag_paired_effect']:+.2f} | {t['flag_W1_col_L1_rank']} | {t['shap']['A_test']['rank']} / {t['shap']['B_test']['rank']} | "
                 f"{t['after_zero_flag']['B_test']['auroc']:.4f} |")
    L.append("\n| P(flag \\| sepsi) | LLM | citato | prima ipotesi | confermato | segnalato sospetto e confermato | posizione nelle ipotesi (per run) |")
    L.append("|---|---|---|---|---|---|---|")
    for p in sorted({r["p"] for r in out["runs"]}, reverse=True):
        for m in LLMS:
            rr = [r for r in out["runs"] if r["p"] == p and r["llm"] == m]
            if not rr:
                continue
            n = len(rr)
            L.append(f"| {p:.2f} | {m} | {sum(r['mentioned'] for r in rr)}/{n} | {sum(r['top_reliance'] for r in rr)}/{n} | "
                     f"{sum(r['confirmed'] for r in rr)}/{n} | {sum(r['suspicious_confirmed'] for r in rr)}/{n} | "
                     + ", ".join(str(r["position"]) if r["position"] else "—" for r in rr) + " |")
    L.append("\n### Ipotesi che citano il flag\n")
    L.append("| P | LLM | run | ipotesi | sospetta | n. feature | esito |")
    L.append("|---|---|---|---|---|---|---|")
    for r in out["runs"]:
        for h in r["hyps"]:
            if h["has_flag"]:
                L.append(f"| {r['p']:.2f} | {r['llm']} | {r['rep']} | {h['title'].replace('|', '/')} | {'sì' if h['suspicious'] else 'no'} | {h['n_features']} | {h['verdict']} |")
    (RES / "phase8.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    {"train": stage_train, "audit": stage_audit, "score": stage_score}[sys.argv[1]]()
