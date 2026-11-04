"""Harness per far lavorare Claude sui pesi: sandbox Python isolata + tool + loop.

La sandbox e' una cartella temporanea che contiene solo:
  - model.npz      (W1, b1, W2, b2)
  - features.json  (nomi e descrizioni degli input, nell'ordine delle colonne di W1)
Il codice dell'LLM gira in un sottoprocesso Python nuovo per ogni chiamata, con cwd = sandbox,
ambiente vuoto (nessuna chiave API), senza rete, e in un mount namespace in cui /home, /root e /tmp
sono sostituiti da cartelle vuote: il codice del progetto, RESULTS.md e i risultati non sono leggibili.

Usato da phase4_audit.py e phase6_correction.py.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import anthropic
import numpy as np

from common import DESCRIPTION

MODEL_ID = "claude-opus-5-5"
EFFORT = "high"
MAX_TURNS = 60
PY_TIMEOUT = 180
MAX_OUT_CHARS = 20000


def make_client():
    api_key = os.environ.get("MY_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Chiave API mancante: imposta MY_ANTHROPIC_API_KEY o ANTHROPIC_API_KEY.")
    # base_url esplicito: nell'ambiente cloud ANTHROPIC_BASE_URL punta a un proxy interno.
    return anthropic.Anthropic(api_key=api_key, base_url="https://api.anthropic.com", max_retries=6,
                               timeout=900)


def make_sandbox(w, feature_names):
    d = Path(tempfile.mkdtemp(prefix="sandbox_", dir="/var/tmp"))
    np.savez(d / "model.npz", W1=w["W1"], b1=w["b1"], W2=w["W2"], b2=np.float64(w["b2"]))
    (d / "features.json").write_text(json.dumps(
        [{"index": i, "name": f, "description": DESCRIPTION[f]} for i, f in enumerate(feature_names)],
        indent=1, ensure_ascii=False))
    return d


def run_python(code, cwd):
    t0 = time.time()
    try:
        p = subprocess.run(["unshare", "--mount", "--map-root-user", "--pid", "--fork", "--mount-proc", "--net",
                            "sh", "-c", "mount -t tmpfs none /home && mount -t tmpfs none /root && "
                            f"mount -t tmpfs none /tmp && cd {cwd} && exec /usr/bin/python3 -I -c \"$0\"", code],
                           cwd=cwd, env={"PATH": "/usr/bin:/bin", "HOME": str(cwd)},
                           capture_output=True, text=True, timeout=PY_TIMEOUT)
        out = p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        rc = p.returncode
    except subprocess.TimeoutExpired:
        out, rc = f"[timeout dopo {PY_TIMEOUT}s]", -1
    if len(out) > MAX_OUT_CHARS:
        out = out[:MAX_OUT_CHARS] + f"\n[output troncato: {len(out)} caratteri in totale]"
    return out or "(nessun output)", rc, time.time() - t0


def forward(w, feature_names, inputs):
    """inputs: lista di dict {feature: valore}; le feature non indicate valgono 0."""
    X = np.zeros((len(inputs), len(feature_names)))
    unknown = set()
    for i, row in enumerate(inputs):
        for f, v in row.items():
            if f in feature_names:
                X[i, feature_names.index(f)] = float(v)
            else:
                unknown.add(f)
    z = X @ w["W1"].T + w["b1"]
    h = np.maximum(z, 0)
    logit = h @ w["W2"] + w["b2"]
    return [{"logit": round(float(l), 4), "probability": round(float(1 / (1 + np.exp(-l))), 6),
             "active_hidden_neurons": [int(j) for j in np.flatnonzero(h[i] > 0)]}
            for i, l in enumerate(logit)], sorted(unknown)


PYTHON_TOOL = {
    "name": "python",
    "description": ("Esegue codice Python 3 (numpy disponibile) in una cartella che contiene model.npz e "
                    "features.json. Ogni chiamata parte da un processo nuovo: lo stato non si conserva tra "
                    "una chiamata e l'altra, a meno che tu non salvi file nella cartella corrente. "
                    "Restituisce stdout e stderr."),
    "input_schema": {"type": "object", "properties": {"code": {"type": "string"}},
                     "required": ["code"], "additionalProperties": False},
}


def forward_tool(budget):
    return {
        "name": "forward_pass",
        "description": (f"Valuta il modello su vettori di input che scegli tu. Budget totale: {budget} vettori "
                        "per tutta l'analisi (ogni vettore conta 1). Ogni vettore e' un oggetto "
                        "{nome_feature: valore}; le feature non indicate valgono 0. Restituisce per ogni "
                        "vettore il logit, la probabilita' e l'elenco dei neuroni nascosti attivi."),
        "input_schema": {"type": "object", "properties": {
            "inputs": {"type": "array", "items": {"type": "object", "additionalProperties": {"type": "number"}}}},
            "required": ["inputs"], "additionalProperties": False},
    }


def run_agent(client, system, user_msg, tools, handlers, submit_name, log_path, max_turns=MAX_TURNS, model=MODEL_ID):
    """Loop manuale. handlers: {tool_name: fn(input) -> (str_result, is_error)}.
    Termina quando l'LLM chiama submit_name con input valido. Salva il transcript in log_path."""
    messages = [{"role": "user", "content": user_msg}]
    log = {"model": model, "effort": EFFORT, "system": system, "user": user_msg, "tools": tools,
           "turns": [], "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}}
    submitted, nudged = None, False
    for turn in range(max_turns):
        with client.messages.stream(model=model, max_tokens=64000, system=system, tools=tools,
                                    messages=messages, thinking={"type": "adaptive"},
                                    output_config={"effort": EFFORT},
                                    cache_control={"type": "ephemeral"}) as stream:
            resp = stream.get_final_message()
        u = resp.usage
        log["usage"]["input"] += u.input_tokens
        log["usage"]["output"] += u.output_tokens
        log["usage"]["cache_read"] += u.cache_read_input_tokens or 0
        log["usage"]["cache_write"] += u.cache_creation_input_tokens or 0
        messages.append({"role": "assistant", "content": resp.content})
        entry = {"turn": turn, "stop_reason": resp.stop_reason, "text": [], "calls": []}
        if resp.stop_reason == "refusal":
            entry["stop_details"] = str(resp.stop_details)
            log["turns"].append(entry)
            break
        results = []
        for b in resp.content:
            if b.type == "text":
                entry["text"].append(b.text)
            elif b.type == "tool_use":
                if b.name == submit_name:
                    submitted = b.input
                    out, err = "Ricevuto.", False
                elif b.name in handlers:
                    out, err = handlers[b.name](b.input)
                else:
                    out, err = f"Tool sconosciuto: {b.name}", True
                entry["calls"].append({"tool": b.name, "input": b.input, "output": out, "is_error": err})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
        log["turns"].append(entry)
        if submitted is not None:
            break
        if results:
            messages.append({"role": "user", "content": results})
        elif resp.stop_reason == "end_turn" and not nudged:
            nudged = True
            messages.append({"role": "user", "content": f"Per concludere chiama il tool {submit_name}."})
        else:
            break
    log["submitted"] = submitted
    log["n_turns"] = len(log["turns"])
    Path(log_path).write_text(json.dumps(log, indent=1, ensure_ascii=False, default=str))
    return submitted, log


def cleanup(d):
    shutil.rmtree(d, ignore_errors=True)
