"""Fase 6c: come la 6b, ma il prompt chiede esplicitamente di poter AGGIUNGERE conoscenza clinica ai pesi
(per esempio riscrivendo neuroni come regole fisiologiche note per la sepsi). Opus 5.5 contro Fable 5.1,
3 run ciascuno, stesse liste di ipotesi, stesso budget.

Oltre a prestazioni e dipendenze si misura il contributo dei neuroni che l'LLM dichiara di aver riscritto:
AUROC del modello senza quei neuroni e AUROC di quei neuroni da soli.

Uso: uv run python phase6c_physiology.py [--models claude-opus-5-5,claude-fable-5-1]
Output: results/phase6c_runs/, models/llm_fixed_phys_*.npz, results/phase6c.json, results/phase6c.md
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import BC, RES, metrics, write_json
from llm_harness import make_client
from phase6_correction import run_one
from phase6b_choice import SOURCES, SUBMIT as SUBMIT_B, prompt_choice
from verify import Ctx

RUNS = RES / "phase6c_runs"
RUNS.mkdir(exist_ok=True)

SUBMIT = json.loads(json.dumps(SUBMIT_B))
SUBMIT["input_schema"]["properties"]["rewritten_neurons"] = {
    "type": "array", "items": {"type": "integer"},
    "description": "Indici dei neuroni nascosti che hai riscritto come regole nuove (vuoto se nessuno)."}
SUBMIT["input_schema"]["required"].append("rewritten_neurons")

EXTRA = """

Oltre a togliere o attenuare meccanismi, puoi anche aggiungere conoscenza clinica ai pesi: per esempio riscrivere alcuni neuroni (riga di W1, b1, peso in W2) come regole fisiologiche note per la sepsi, scegliendo tu quali neuroni sacrificare. Se lo fai, indica in submit_fix quali neuroni hai riscritto."""


def prompt_phys(hyps, n_in, budget):
    system, user = prompt_choice(hyps, n_in, budget)
    return system, user + EXTRA


def contributions(ctx, w, neurons):
    out = {}
    for s in ("A_test", "B_test"):
        X, y = ctx.X[s], ctx.y[s]
        h = np.maximum(X @ w["W1"].T + w["b1"], 0)
        full = h @ w["W2"]
        part = h[:, neurons] @ w["W2"][neurons] if neurons else np.zeros(len(X))
        out[s] = {"without_rewritten": metrics(y, full - part)["auroc"],
                  "rewritten_only": metrics(y, part)["auroc"] if neurons and np.std(part) > 0 else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="claude-opus-5-5,claude-fable-5-1")
    ap.add_argument("--budget", type=int, default=30)
    a = ap.parse_args()
    ctx = Ctx()
    Ctx.cached = ctx
    p5 = json.loads((RES / "phase5.json").read_text())
    p4 = {(r["cond"], r["rep"]): r for r in json.loads((RES / "phase4.json").read_text())["runs"]}
    lists = {}
    for cond, rep in SOURCES:
        v = next(r for r in p5["runs"] if r["cond"] == cond and r["rep"] == rep)
        ok = {h["id"] for h in v["hypotheses"] if h["suspicious"] and h["verdict"] == "confermata"}
        lists[(cond, rep)] = [{k: h[k] for k in ("id", "title", "description", "features", "neurons")}
                              for h in p4[(cond, rep)]["report"]["hypotheses"] if h["id"] in ok]
    client = make_client()
    jobs = [(m, c, r) for m in a.models.split(",") for c, r in SOURCES]

    def go(j):
        m, c, r = j
        tag = "phys_" + m.replace("claude-", "").replace("-", "") + "_"
        res = run_one(c, r, lists[(c, r)], a.budget, client, prompt_fn=prompt_phys, submit=SUBMIT,
                      tag=tag, model=m, runs_dir=RUNS)
        res["model"] = m
        return res

    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(go, jobs))
    w0 = ctx.w0
    zero_bc = {**w0, "W1": w0["W1"].copy()}
    zero_bc["W1"][:, ctx.feats.index(BC)] = 0
    out = {"original": ctx.profile(w0), "reference_zero_bc_column": ctx.profile(zero_bc), "fixes": []}
    for r in res:
        rr = {k: v for k, v in r.items() if k != "w"}
        if r.get("w") is not None:
            rr["profile"] = ctx.profile(r["w"])
            neu = [n for n in (r["submitted"] or {}).get("rewritten_neurons", []) if 0 <= n < 32]
            # controllo indipendente: neuroni la cui riga di W1 e' cambiata in modo non spiegabile da
            # operazioni per colonna (scala/azzeramento di colonne) -> confronto riga per riga
            d = np.abs(r["w"]["W1"] - w0["W1"]).sum(1) + np.abs(r["w"]["W2"] - w0["W2"])
            rr["rewritten_neurons_declared"] = neu
            rr["neurons_W2_changed"] = [int(j) for j in np.flatnonzero(np.abs(r["w"]["W2"] - w0["W2"]) > 1e-9)]
            rr["contrib"] = contributions(ctx, r["w"], neu)
            rr["orig_without_those_neurons"] = contributions(ctx, w0, neu)
        out["fixes"].append(rr)
    write_json("phase6c.json", out)
    write_md(out)


def write_md(out):
    L = ["| modello | lista | A AUROC | B AUROC | B AUPRC | neuroni riscritti | B senza neuroni riscritti | B solo neuroni riscritti | A solo neuroni riscritti | bc | conteggi | ICULOS | Unit_SICU |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, p in (("originale", out["original"]), ("solo colonna bc azzerata", out["reference_zero_bc_column"])):
        L.append(f"| {name} | — | {p['A_test']['auroc']:.4f} | {p['B_test']['auroc']:.4f} | {p['B_test']['auprc']:.4f} | — | — | — | — | "
                 f"{p['dep']['bc_paired_do1_minus_do0']:+.3f} | {p['dep']['counts_abs_dlogit_to_median']:.3f} | {p['dep']['ICULOS_abs_dlogit_to_median']:.3f} | {p['dep']['Unit_SICU_abs_dlogit_to_median']:.3f} |")
    for f in out["fixes"]:
        if "profile" not in f:
            L.append(f"| {f['model']} | rep{f['rep']} | pesi non consegnati |||||||||||")
            continue
        p, c = f["profile"], f["contrib"]
        fmt = lambda v: "—" if v is None else f"{v:.4f}"
        L.append(f"| {f['model']} | rep{f['rep']} | {p['A_test']['auroc']:.4f} | {p['B_test']['auroc']:.4f} | {p['B_test']['auprc']:.4f} | "
                 f"{f['rewritten_neurons_declared'] or 'nessuno'} | {fmt(c['B_test']['without_rewritten'])} | {fmt(c['B_test']['rewritten_only'])} | {fmt(c['A_test']['rewritten_only'])} | "
                 f"{p['dep']['bc_paired_do1_minus_do0']:+.3f} | {p['dep']['counts_abs_dlogit_to_median']:.3f} | {p['dep']['ICULOS_abs_dlogit_to_median']:.3f} | {p['dep']['Unit_SICU_abs_dlogit_to_median']:.3f} |")
    for f in out["fixes"]:
        L.append(f"\n**{f['model']} (rep{f['rep']})**\n")
        if f.get("submitted"):
            L.append("| meccanismo | decisione | motivazione |")
            L.append("|---|---|---|")
            for d in f["submitted"]["decisions"]:
                L.append(f"| {d['mechanism'].replace('|', '/')} | **{d['decision']}** | {d['reason'].replace('|', '/')} |")
            L.append(f"\nModifiche: {f['submitted']['changes']}\n")
        if "neurons_W2_changed" in f:
            L.append(f"Neuroni con W2 modificato (controllo): {f['neurons_W2_changed']}; forward pass usati: {f['forward_used']}; turni: {f['n_turns']}\n")
    (RES / "phase6c.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
