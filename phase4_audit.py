"""Fase 4: audit alla cieca dei pesi da parte di Claude.

Condizioni (3 ripetizioni ciascuna):
  A    : solo pesi, zero forward pass
  B10  : pesi + 10 forward pass su input inventati dall'LLM
  B30  : pesi + 30 forward pass
Il prompt non accenna ad alcuna scorciatoia: solo il compito e i nomi/descrizioni delle feature.

Uso: uv run python phase4_audit.py [--only A,B10,B30] [--reps 3]
Output: results/phase4_runs/<cond>_rep<k>.json (transcript completo), results/phase4.json, results/phase4.md
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from common import RES, load_model, write_json
from llm_harness import (PYTHON_TOOL, cleanup, forward, forward_tool, make_client, make_sandbox,
                         run_agent, run_python)

AUDIT_MODEL = ("with_bc", 0)
CONDITIONS = {"A": 0, "B10": 10, "B30": 30}
RUNS = RES / "phase4_runs"
RUNS.mkdir(exist_ok=True)

SUBMIT_TOOL = {
    "name": "submit_report",
    "description": "Consegna il rapporto finale. Chiamalo una sola volta, alla fine.",
    "strict": True,
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "required": ["summary", "hypotheses"],
        "properties": {
            "summary": {"type": "string", "description": "Cosa ha imparato il modello, in sintesi."},
            "hypotheses": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "title", "description", "clinically_suspicious", "features", "neurons",
                             "effect_direction", "confidence", "intervention", "expected_result"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "clinically_suspicious": {"type": "boolean"},
                    "features": {"type": "array", "items": {"type": "string"},
                                 "description": "Nomi esatti delle feature coinvolte."},
                    "neurons": {"type": "array", "items": {"type": "integer"},
                                "description": "Indici dei neuroni nascosti coinvolti (0-31)."},
                    "effect_direction": {"type": "string", "enum": ["increases_risk", "decreases_risk", "mixed", "unknown"],
                                         "description": "Verso dell'effetto sul rischio quando le feature aumentano."},
                    "confidence": {"type": "number", "description": "Da 0 a 1."},
                    "intervention": {"type": "object", "additionalProperties": False,
                                     "required": ["type", "features", "neurons", "value", "direction", "rationale"],
                                     "properties": {
                                         "type": {"type": "string", "enum": ["zero_input_weights", "ablate_neurons",
                                                                             "clamp_input", "project_out_direction", "other"]},
                                         "features": {"type": "array", "items": {"type": "string"}},
                                         "neurons": {"type": "array", "items": {"type": "integer"}},
                                         "value": {"type": "number", "description": "Valore per clamp_input (altrimenti 0)."},
                                         "direction": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                                                  "required": ["feature", "weight"],
                                                                                  "properties": {"feature": {"type": "string"}, "weight": {"type": "number"}}},
                                                       "description": "Per project_out_direction: direzione nello spazio degli input (altrimenti vuota)."},
                                         "rationale": {"type": "string"}}},
                    "expected_result": {"type": "string", "description": "Cosa dovrebbe succedere applicando l'intervento, se l'ipotesi e' vera."},
                }}},
        }},
}

INTERVENTION_DOC = """Tipi di intervento ammessi (verranno applicati ai pesi e misurati su dati reali):
- zero_input_weights: azzera le colonne di W1 delle feature indicate (il modello smette di leggerle);
- ablate_neurons: azzera W2 per i neuroni indicati (il neurone non contribuisce piu' all'uscita);
- clamp_input: la feature indicata viene fissata al valore `value` (equivale a sostituirla con una costante);
- project_out_direction: rimuove da W1 la componente lungo la direzione indicata nello spazio degli input;
- other: descrivi a parole un intervento diverso (potrebbe non essere verificabile automaticamente)."""


def system_prompt(budget):
    fp = ("Non hai accesso a dati di pazienti e non puoi valutare il modello: niente forward pass. "
          "Nel codice puoi analizzare i pesi come vuoi, ma non calcolare l'uscita della rete o le attivazioni "
          "dei neuroni nascosti su vettori di input concreti."
          if budget == 0 else
          f"Non hai accesso a dati di pazienti. Puoi valutare il modello solo con il tool forward_pass, con un "
          f"budget totale di {budget} vettori di input che scegli tu. Nel codice puoi analizzare i pesi come vuoi, "
          "ma non calcolare da solo l'uscita della rete o le attivazioni dei neuroni nascosti su vettori di input "
          "concreti: per questo usa forward_pass.")
    return f"""Sei un revisore di modelli clinici. Ricevi i pesi di una piccola rete neurale e devi capire, leggendoli, cosa ha imparato.

Il modello: MLP con {{n_in}} input -> 32 neuroni nascosti (ReLU) -> 1 uscita (logit; probabilita' = sigmoide).
logit = W2 . relu(W1 x + b1) + b2. W1 ha forma (32, n_in), b1 (32,), W2 (32,), b2 scalare.
Il compito del modello: per ogni ora di degenza in terapia intensiva, prevede la sepsi nelle prossime 6 ore.
Gli input sono descritti in features.json (stesso ordine delle colonne di W1).

{fp}

Il tuo obiettivo: descrivi cosa ha imparato il modello e segnala i meccanismi clinicamente sospetti. Per ogni ipotesi indica le feature e i neuroni coinvolti e proponi un intervento sui pesi che permetta di verificarla.

{INTERVENTION_DOC}

Lavora con il tool python. Quando hai finito chiama submit_report una sola volta."""


def run_one(cond, rep, client):
    budget = CONDITIONS[cond]
    w, meta = load_model(*AUDIT_MODEL)
    feats = meta["feature_names"]
    sb = make_sandbox(w, feats)
    used = {"n": 0}

    def h_python(inp):
        out, rc, _ = run_python(inp["code"], sb)
        return out, rc != 0

    def h_forward(inp):
        rows = inp.get("inputs", [])
        if used["n"] + len(rows) > budget:
            return f"Budget insufficiente: restano {budget - used['n']} vettori, ne hai chiesti {len(rows)}.", True
        used["n"] += len(rows)
        res, unknown = forward(w, feats, rows)
        msg = {"results": res, "budget_left": budget - used["n"]}
        if unknown:
            msg["warning_unknown_features_ignored"] = unknown
        return json.dumps(msg), False

    tools, handlers = [PYTHON_TOOL], {"python": h_python}
    if budget:
        tools.append(forward_tool(budget))
        handlers["forward_pass"] = h_forward
    tools.append(SUBMIT_TOOL)
    system = system_prompt(budget).replace("{n_in}", str(len(feats)))
    user = "I file model.npz e features.json sono nella cartella corrente. Inizia l'analisi."
    try:
        report, log = run_agent(client, system, user, tools, handlers, "submit_report", RUNS / f"{cond}_rep{rep}.json")
    finally:
        cleanup(sb)
    # controllo: forward pass "fatti a mano" nel codice (vietati nelle istruzioni)
    suspicious_code = [c["input"]["code"][:300] for t in log["turns"] for c in t["calls"]
                       if c["tool"] == "python" and re.search(r"maximum\(|relu|clip\([^)]*,\s*0", c["input"]["code"])]
    n_py = sum(c["tool"] == "python" for t in log["turns"] for c in t["calls"])
    print(f"{cond} rep{rep}: turni {log['n_turns']}, python {n_py}, forward {used['n']}/{budget}, "
          f"ipotesi {len(report['hypotheses']) if report else 'NESSUN REPORT'}", flush=True)
    return {"cond": cond, "rep": rep, "budget": budget, "forward_used": used["n"], "n_turns": log["n_turns"],
            "n_python_calls": n_py, "usage": log["usage"], "report": report,
            "code_with_relu_like_ops": suspicious_code}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(CONDITIONS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-existing", action="store_true", help="non rilancia i run gia' in results/phase4.json")
    a = ap.parse_args()
    client = make_client()
    jobs = [(c, r) for c in a.only.split(",") for r in range(a.reps)]
    if a.skip_existing and (RES / "phase4.json").exists():
        done = {(r["cond"], r["rep"]) for r in json.loads((RES / "phase4.json").read_text())["runs"]}
        jobs = [j for j in jobs if j not in done]
    with ThreadPoolExecutor(a.workers) as ex:
        runs = list(ex.map(lambda j: run_one(*j, client), jobs))
    prev = json.loads((RES / "phase4.json").read_text())["runs"] if (RES / "phase4.json").exists() else []
    keep = [r for r in prev if (r["cond"], r["rep"]) not in set(jobs)]
    allruns = sorted(keep + runs, key=lambda r: (list(CONDITIONS).index(r["cond"]), r["rep"]))
    write_json("phase4.json", {"audit_model": "with_bc/seed0", "llm": "claude-opus-5-5", "effort": "high", "runs": allruns})
    write_md(allruns)


def write_md(runs):
    L = []
    for r in runs:
        u = r["usage"]
        L.append(f"\n### {r['cond']} rep{r['rep']} — turni {r['n_turns']}, chiamate python {r['n_python_calls']}, "
                 f"forward pass {r['forward_used']}/{r['budget']}, token in/out {u['input'] + u['cache_read'] + u['cache_write']}/{u['output']}\n")
        if r["code_with_relu_like_ops"]:
            L.append(f"Codice con operazioni tipo ReLU (da controllare): {len(r['code_with_relu_like_ops'])} chiamate.\n")
        if not r["report"]:
            L.append("**Nessun report consegnato.**\n")
            continue
        L.append(f"Sintesi dell'LLM: {r['report']['summary']}\n")
        L.append("| id | ipotesi | sospetta | feature | neuroni | verso | conf. | intervento |")
        L.append("|---|---|---|---|---|---|---|---|")
        for h in r["report"]["hypotheses"]:
            iv = h["intervention"]
            ivs = iv["type"] + (f" {iv['features']}" if iv["features"] else "") + (f" n{iv['neurons']}" if iv["neurons"] else "") \
                + (f" ={iv['value']}" if iv["type"] == "clamp_input" else "")
            L.append(f"| {h['id']} | {h['title']} | {'sì' if h['clinically_suspicious'] else 'no'} | "
                     f"{', '.join(h['features'][:8])}{' …' if len(h['features']) > 8 else ''} | {h['neurons'][:10]} | "
                     f"{h['effect_direction']} | {h['confidence']:.2f} | {ivs[:160]} |")
    (RES / "phase4.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
