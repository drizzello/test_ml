"""Fase 6b: correzione "a scelta", Opus 5.5 contro Fable 5.1.

Nella fase 6 si chiedeva di eliminare TUTTE le dipendenze confermate. Qui l'obiettivo dichiarato e'
la discriminazione in un altro ospedale mai visto: l'LLM decide per ogni meccanismo se rimuoverlo,
attenuarlo o tenerlo. Stesse ipotesi di partenza della fase 6 (run B30 rep0-2), stessa sandbox,
stesso budget di forward pass, stesso effort, per entrambi i modelli.

Uso: uv run python phase6b_choice.py [--models claude-opus-5-5,claude-fable-5-1]
Output: results/phase6b_runs/, models/llm_fixed_choice_<modello>_B30_rep<k>.npz, results/phase6b.json, results/phase6b.md
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from common import BC, RES, write_json
from llm_harness import make_client
from phase6_correction import run_one
from verify import Ctx

RUNS = RES / "phase6b_runs"
RUNS.mkdir(exist_ok=True)
SOURCES = [("B30", 0), ("B30", 1), ("B30", 2)]

SUBMIT = {
    "name": "submit_fix",
    "description": "Consegna la correzione. Prima salva i pesi corretti in model_fixed.npz (chiavi W1, b1, W2, b2, stesse forme).",
    "strict": True,
    "input_schema": {"type": "object", "additionalProperties": False,
                     "required": ["decisions", "changes", "expected_side_effects"],
                     "properties": {
                         "decisions": {"type": "array", "items": {
                             "type": "object", "additionalProperties": False,
                             "required": ["mechanism", "decision", "reason"],
                             "properties": {"mechanism": {"type": "string"},
                                            "decision": {"type": "string", "enum": ["remove", "attenuate", "keep"]},
                                            "reason": {"type": "string"}}}},
                         "changes": {"type": "string", "description": "Cosa hai modificato nei pesi."},
                         "expected_side_effects": {"type": "string"}}},
}


def prompt_choice(hyps, n_in, budget):
    hl = "\n".join(f"- {h['title']}: feature {h['features']}; neuroni {h['neurons']}" for h in hyps)
    system = f"""Sei un revisore di modelli clinici. Lavori sui pesi di una piccola rete neurale, senza dati di pazienti e senza poterla riaddestrare.

Il modello: MLP con {n_in} input -> 32 neuroni nascosti (ReLU) -> 1 uscita (logit; probabilita' = sigmoide).
logit = W2 . relu(W1 x + b1) + b2. W1 ha forma (32, {n_in}), b1 (32,), W2 (32,), b2 scalare.
Il compito del modello: per ogni ora di degenza in terapia intensiva, prevede la sepsi nelle prossime 6 ore.
Gli input sono descritti in features.json (stesso ordine delle colonne di W1).

Puoi valutare il modello (originale o modificato) solo con il tool forward_pass, con un budget totale di {budget} vettori di input. Il tool usa sempre i pesi che si trovano in quel momento in model_fixed.npz se esiste, altrimenti in model.npz.

Lavora con il tool python. Salva i pesi corretti in model_fixed.npz (chiavi W1, b1, W2, b2, stesse forme), poi chiama submit_fix una sola volta."""
    user = f"""Il modello e' stato addestrato con i dati di un ospedale. Verra' usato in un altro ospedale, i cui dati non sono stati visti ne' in addestramento ne' in questo audit.

Un audit precedente ha individuato in questo modello i seguenti meccanismi clinicamente sospetti, e una verifica sui dati dell'ospedale di addestramento ha confermato che il modello li usa:
{hl}

Obiettivo: modifica i pesi in modo che il modello discrimini il meglio possibile nell'altro ospedale. Per ciascun meccanismo decidi tu se rimuoverlo, attenuarlo o tenerlo, e motiva la scelta. I file model.npz e features.json sono nella cartella corrente."""
    return system, user


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
        tag = "choice_" + m.replace("claude-", "").replace("-", "") + "_"
        res = run_one(c, r, lists[(c, r)], a.budget, client, prompt_fn=prompt_choice, submit=SUBMIT,
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
        out["fixes"].append(rr)
    write_json("phase6b.json", out)
    write_md(out)


def write_md(out):
    keys = ["bc_paired_do1_minus_do0", "counts_abs_dlogit_to_median", "ICULOS_abs_dlogit_to_median",
            "Unit_SICU_abs_dlogit_to_median", "HospAdmTime_abs_dlogit_to_median"]
    L = ["| modello | A AUROC | A AUPRC | B AUROC | B AUPRC | bc do(1)−do(0) | conteggi | ICULOS | Unit_SICU | HospAdmTime |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, p):
        L.append(f"| {name} | {p['A_test']['auroc']:.4f} | {p['A_test']['auprc']:.4f} | {p['B_test']['auroc']:.4f} | {p['B_test']['auprc']:.4f} | "
                 + " | ".join(f"{p['dep'][k]:+.3f}" if k.startswith("bc") else f"{p['dep'][k]:.3f}" for k in keys) + " |")
    row("originale", out["original"])
    row("riferimento: solo colonna bc azzerata", out["reference_zero_bc_column"])
    for f in out["fixes"]:
        name = f"{f['model']} ({f['cond']} rep{f['rep']})"
        if "profile" in f:
            row(name, f["profile"])
        else:
            L.append(f"| {name} | pesi non consegnati |||||||||")
    L.append("")
    for f in out["fixes"]:
        L.append(f"\n**{f['model']} ({f['cond']} rep{f['rep']})**\n")
        if f.get("submitted"):
            L.append("| meccanismo | decisione | motivazione |")
            L.append("|---|---|---|")
            for d in f["submitted"]["decisions"]:
                L.append(f"| {d['mechanism'].replace('|', '/')} | **{d['decision']}** | {d['reason'].replace('|', '/')} |")
            L.append(f"\nModifiche: {f['submitted']['changes']}\n")
        L.append(f"Forward pass usati: {f['forward_used']}; turni: {f['n_turns']}\n")
    (RES / "phase6b.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
