"""Fase 6: correzione dei pesi da parte di Claude, senza dati e senza riaddestramento.

Per ogni run scelto della fase 4 si prendono le ipotesi segnalate come sospette e confermate nella
fase 5. In una nuova conversazione, con la stessa sandbox (pesi + nomi feature) e un budget di
forward pass, si chiede a Claude di modificare i pesi per eliminare quelle dipendenze preservando il
resto. Claude salva model_fixed.npz nella sandbox; il modello corretto si misura su A-test e B.

Uso: uv run python phase6_correction.py [--runs B30:0,B30:1,B30:2] [--budget 30]
Output: results/phase6_runs/<cond>_rep<k>.json, models/llm_fixed_<cond>_rep<k>.npz,
        results/phase6.json, results/phase6.md
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import BC, MODELS, RES, write_json
from llm_harness import (PYTHON_TOOL, cleanup, forward, forward_tool, make_client, make_sandbox,
                         run_agent, run_python)
from verify import Ctx

RUNS = RES / "phase6_runs"
RUNS.mkdir(exist_ok=True)

SUBMIT = {
    "name": "submit_fix",
    "description": "Consegna la correzione. Prima salva i pesi corretti in model_fixed.npz (chiavi W1, b1, W2, b2, stesse forme).",
    "strict": True,
    "input_schema": {"type": "object", "additionalProperties": False, "required": ["changes", "expected_side_effects"],
                     "properties": {"changes": {"type": "string", "description": "Cosa hai modificato e perche'."},
                                    "expected_side_effects": {"type": "string"}}},
}


def prompt(hyps, n_in, budget):
    hl = "\n".join(f"- {h['title']}: feature {h['features']}; neuroni {h['neurons']}" for h in hyps)
    system = f"""Sei un revisore di modelli clinici. Lavori sui pesi di una piccola rete neurale, senza dati di pazienti e senza poterla riaddestrare.

Il modello: MLP con {n_in} input -> 32 neuroni nascosti (ReLU) -> 1 uscita (logit; probabilita' = sigmoide).
logit = W2 . relu(W1 x + b1) + b2. W1 ha forma (32, {n_in}), b1 (32,), W2 (32,), b2 scalare.
Il compito del modello: per ogni ora di degenza in terapia intensiva, prevede la sepsi nelle prossime 6 ore.
Gli input sono descritti in features.json (stesso ordine delle colonne di W1).

Puoi valutare il modello (originale o modificato) solo con il tool forward_pass, con un budget totale di {budget} vettori di input. Il tool usa sempre i pesi che si trovano in quel momento in model_fixed.npz se esiste, altrimenti in model.npz.

Lavora con il tool python. Salva i pesi corretti in model_fixed.npz (chiavi W1, b1, W2, b2, stesse forme), poi chiama submit_fix una sola volta."""
    user = f"""Un audit precedente ha individuato in questo modello i seguenti meccanismi clinicamente sospetti, e una verifica su dati reali li ha confermati:
{hl}

Modifica i pesi per eliminare la dipendenza del modello da ciascuno di questi meccanismi, preservando il piu' possibile il resto del comportamento. I file model.npz e features.json sono nella cartella corrente."""
    return system, user


def run_one(cond, rep, hyps, budget, client, prompt_fn=None, submit=None, tag="", model=None, runs_dir=None):
    ctx_w0 = Ctx.cached.w0
    feats = Ctx.cached.feats
    sb = make_sandbox(ctx_w0, feats)
    used = {"n": 0}

    def current_w():
        p = sb / "model_fixed.npz"
        if p.exists():
            try:
                d = np.load(p)
                return {"W1": d["W1"], "b1": d["b1"], "W2": d["W2"], "b2": float(d["b2"])}
            except Exception:
                pass
        return ctx_w0

    def h_python(inp):
        out, rc, _ = run_python(inp["code"], sb)
        return out, rc != 0

    def h_forward(inp):
        rows = inp.get("inputs", [])
        if used["n"] + len(rows) > budget:
            return f"Budget insufficiente: restano {budget - used['n']} vettori, ne hai chiesti {len(rows)}.", True
        used["n"] += len(rows)
        res, unknown = forward(current_w(), feats, rows)
        return json.dumps({"results": res, "budget_left": budget - used["n"], "unknown_features_ignored": unknown}), False

    system, user = (prompt_fn or prompt)(hyps, len(feats), budget)
    tools = [PYTHON_TOOL, forward_tool(budget), submit or SUBMIT]
    extra = {"model": model} if model else {}
    try:
        sub, log = run_agent(client, system, user, tools, {"python": h_python, "forward_pass": h_forward},
                             "submit_fix", (runs_dir or RUNS) / f"{tag}{cond}_rep{rep}.json", **extra)
        p = sb / "model_fixed.npz"
        w1 = None
        if p.exists():
            d = np.load(p)
            w1 = {"W1": d["W1"], "b1": d["b1"], "W2": d["W2"], "b2": float(d["b2"])}
            ok = w1["W1"].shape == ctx_w0["W1"].shape and w1["W2"].shape == ctx_w0["W2"].shape and w1["b1"].shape == ctx_w0["b1"].shape
            if not ok:
                w1 = None
            else:
                np.savez(MODELS / f"llm_fixed_{tag}{cond}_rep{rep}.npz", **w1)
    finally:
        cleanup(sb)
    print(f"fix {tag}{cond} rep{rep}: turni {log['n_turns']}, forward {used['n']}, pesi {'OK' if w1 else 'MANCANTI'}", flush=True)
    return {"cond": cond, "rep": rep, "hypotheses": hyps, "submitted": sub, "forward_used": used["n"],
            "n_turns": log["n_turns"], "usage": log["usage"], "w": w1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="B30:0,B30:1,B30:2")
    ap.add_argument("--budget", type=int, default=30)
    a = ap.parse_args()
    ctx = Ctx()
    Ctx.cached = ctx
    p5 = json.loads((RES / "phase5.json").read_text())
    p4 = {(r["cond"], r["rep"]): r for r in json.loads((RES / "phase4.json").read_text())["runs"]}
    jobs = []
    for spec in a.runs.split(","):
        cond, rep = spec.split(":")
        rep = int(rep)
        v = next(r for r in p5["runs"] if r["cond"] == cond and r["rep"] == rep)
        ok_ids = {h["id"] for h in v["hypotheses"] if h["suspicious"] and h["verdict"] == "confermata"}
        hyps = [{k: h[k] for k in ("id", "title", "description", "features", "neurons")}
                for h in p4[(cond, rep)]["report"]["hypotheses"] if h["id"] in ok_ids]
        jobs.append((cond, rep, hyps))
    client = make_client()
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(lambda j: run_one(*j, a.budget, client) if j[2] else
                          {"cond": j[0], "rep": j[1], "hypotheses": [], "skipped": "nessuna ipotesi sospetta confermata", "w": None}, jobs))
    # riferimenti
    w0 = ctx.w0
    zero_bc = {**w0, "W1": w0["W1"].copy()}
    zero_bc["W1"][:, ctx.feats.index(BC)] = 0
    p2 = json.loads((RES / "phase2.json").read_text())
    out = {"original": ctx.profile(w0), "reference_zero_bc_column": ctx.profile(zero_bc),
           "reference_retrained_no_bc_seed0": {s: p2["runs"]["no_bc/seed0"][s] for s in ("A_test", "B_test")},
           "fixes": []}
    for r in res:
        rr = {k: v for k, v in r.items() if k != "w"}
        if r.get("w") is not None:
            rr["profile"] = ctx.profile(r["w"])
            rr["weight_change_rel_fro"] = float(np.sqrt(sum(np.sum((np.asarray(r["w"][k]) - np.asarray(w0[k])) ** 2) for k in ("W1", "b1", "W2")))
                                                / np.sqrt(sum(np.sum(np.asarray(w0[k]) ** 2) for k in ("W1", "b1", "W2"))))
        out["fixes"].append(rr)
    write_json("phase6.json", out)
    write_md(out)


def write_md(out):
    keys = ["bc_paired_do1_minus_do0", "counts_abs_dlogit_to_median", "ICULOS_abs_dlogit_to_median",
            "ICULOS_slope_168h_minus_24h", "Unit_SICU_abs_dlogit_to_median", "physiology_last_abs_dlogit_to_median"]
    L = ["| modello | A AUROC | A AUPRC | B AUROC | B AUPRC | bc do(1)-do(0) | conteggi | ICULOS | pendenza ICULOS 24→168h | Unit_SICU | fisiologia |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, p):
        L.append(f"| {name} | {p['A_test']['auroc']:.4f} | {p['A_test']['auprc']:.4f} | {p['B_test']['auroc']:.4f} | {p['B_test']['auprc']:.4f} | "
                 + " | ".join(f"{p['dep'][k]:+.3f}" if k.startswith(("bc", "ICULOS_slope")) else f"{p['dep'][k]:.3f}" for k in keys) + " |")
    row("originale (with_bc seed 0)", out["original"])
    row("riferimento: colonna bc azzerata", out["reference_zero_bc_column"])
    nb = out["reference_retrained_no_bc_seed0"]
    L.append(f"| riferimento: riaddestrato senza bc (seed 0) | {nb['A_test']['auroc']:.4f} | {nb['A_test']['auprc']:.4f} | {nb['B_test']['auroc']:.4f} | {nb['B_test']['auprc']:.4f} | — | — | — | — | — | — |")
    for f in out["fixes"]:
        if "profile" in f:
            row(f"correzione LLM {f['cond']} rep{f['rep']}", f["profile"])
        else:
            L.append(f"| correzione LLM {f['cond']} rep{f['rep']} | {f.get('skipped', 'pesi non consegnati')} |||||||||||")
    L.append("")
    for f in out["fixes"]:
        L.append(f"**{f['cond']} rep{f['rep']}** — ipotesi da correggere: " + "; ".join(h["title"] for h in f["hypotheses"]))
        if f.get("submitted"):
            L.append(f"\nModifiche dichiarate dall'LLM: {f['submitted']['changes']}\n\nEffetti collaterali attesi: {f['submitted']['expected_side_effects']}\n")
        if "weight_change_rel_fro" in f:
            L.append(f"Variazione relativa dei pesi (norma di Frobenius): {f['weight_change_rel_fro']:.3f}; forward pass usati: {f['forward_used']}\n")
    (RES / "phase6.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
