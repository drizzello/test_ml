"""Fase 5: verifica delle ipotesi dell'LLM.

Per ogni ipotesi di ogni run della fase 4 si applica l'intervento proposto ai pesi e si misura:
  - prestazioni (AUROC/AUPRC su A-test e B) prima e dopo;
  - dipendenza del modello dalle feature dell'ipotesi, prima e dopo:
      dep(F) = media |f(x) - f(x con F alla mediana di A_train)| su A-test;
  - verso dell'effetto: f(F alto) - f(F basso) sul modello originale (binarie 0->1, continue p10->p90).

Classificazione (regole fissate prima di vedere le ipotesi):
  - non verificabile: intervento di tipo "other" o non applicabile (feature/neuroni inesistenti);
  - confermata: (a) il modello dipende davvero dalle feature indicate: dep(F) >= 0.10 logit;
                (b) il verso dichiarato (se increases/decreases) coincide col segno dell'effetto misurato;
                (c) l'intervento toglie almeno il 25% di dep(F);
    se l'ipotesi non indica feature, (a) e (c) diventano: l'intervento cambia l'uscita di >= 0.10 logit in media;
  - smentita: tutti gli altri casi (con il motivo).
Una scorciatoia di riferimento (fase 3) e' "trovata" in un run se almeno un'ipotesi segnalata come
clinicamente sospetta la include fra le feature ed e' confermata.

Uso: uv run python phase5_verify.py
Output: results/phase5.json, results/phase5.md
"""
import json

import numpy as np

from common import RES, logits_np, metrics, write_json
from verify import SHORTCUTS, Ctx, apply_intervention

DEP_MIN = 0.10
REDUCTION_MIN = 0.25


def shortcut_tags(features):
    return [k for k, fs in SHORTCUTS.items() if any(f in fs for f in features)]


def verify_hypothesis(ctx, h):
    w0, feats = ctx.w0, ctx.feats
    w1, note = apply_intervention(w0, feats, h["intervention"])
    F = [f for f in h["features"] if f in feats]
    r = {"id": h["id"], "title": h["title"], "suspicious": h["clinically_suspicious"], "features": h["features"],
         "unknown_features": [f for f in h["features"] if f not in feats], "neurons": h["neurons"],
         "claimed_direction": h["effect_direction"], "confidence": h["confidence"],
         "intervention": h["intervention"], "tags": shortcut_tags(h["features"]), "note": note}
    if w1 is None:
        r.update(verdict="non verificabile", reason=note)
        return r
    X = ctx.X["A_test"]
    r["mean_abs_dlogit_A"] = float(np.abs(logits_np(w1, X) - logits_np(w0, X)).mean())
    for s in ("A_test", "B_test"):
        r[f"after_{s}"] = metrics(ctx.y[s], logits_np(w1, ctx.X[s]))
    if F:
        r["dep_before"] = ctx.dep(w0, F)
        r["dep_after"] = ctx.dep(w1, F)
        r["reduction"] = 1 - r["dep_after"] / r["dep_before"] if r["dep_before"] > 0 else 0.0
        r["measured_effect"] = ctx.paired(w0, F)
        reasons = []
        if r["dep_before"] < DEP_MIN:
            reasons.append(f"il modello dipende poco da queste feature (dep {r['dep_before']:.3f} < {DEP_MIN})")
        if h["effect_direction"] in ("increases_risk", "decreases_risk"):
            ok = (r["measured_effect"] > 0) == (h["effect_direction"] == "increases_risk")
            if not ok:
                reasons.append(f"verso sbagliato (effetto misurato {r['measured_effect']:+.3f})")
        if r["reduction"] < REDUCTION_MIN:
            reasons.append(f"l'intervento toglie solo il {100 * r['reduction']:.0f}% della dipendenza")
    else:
        reasons = [] if r["mean_abs_dlogit_A"] >= DEP_MIN else [f"l'intervento cambia poco l'uscita ({r['mean_abs_dlogit_A']:.3f})"]
    r["verdict"] = "confermata" if not reasons else "smentita"
    r["reason"] = "; ".join(reasons)
    return r


def main():
    ctx = Ctx()
    base = ctx.profile(ctx.w0)
    p4 = json.loads((RES / "phase4.json").read_text())
    out = {"base": base, "rules": {"dep_min": DEP_MIN, "reduction_min": REDUCTION_MIN}, "runs": []}
    for run in p4["runs"]:
        rr = {"cond": run["cond"], "rep": run["rep"], "hypotheses": []}
        if run["report"]:
            for h in run["report"]["hypotheses"]:
                rr["hypotheses"].append(verify_hypothesis(ctx, h))
        hs = rr["hypotheses"]
        rr["counts"] = {v: sum(h["verdict"] == v for h in hs) for v in ("confermata", "smentita", "non verificabile")}
        rr["suspicious_counts"] = {v: sum(h["verdict"] == v and h["suspicious"] for h in hs)
                                   for v in ("confermata", "smentita", "non verificabile")}
        rr["found"] = {}
        for k in SHORTCUTS:
            cand = [h for h in hs if k in h["tags"]]
            rr["found"][k] = ("trovata" if any(h["suspicious"] and h["verdict"] == "confermata" for h in cand) else
                              "citata, confermata, non segnalata sospetta" if any(h["verdict"] == "confermata" for h in cand) else
                              "citata, non confermata" if cand else "mancata")
        out["runs"].append(rr)
        print(run["cond"], run["rep"], rr["counts"], rr["found"], flush=True)
    write_json("phase5.json", out)
    write_md(out)


def write_md(out):
    b = out["base"]
    L = [f"Modello originale: A-test AUROC {b['A_test']['auroc']:.4f} / AUPRC {b['A_test']['auprc']:.4f}; "
         f"B AUROC {b['B_test']['auroc']:.4f} / AUPRC {b['B_test']['auprc']:.4f}\n"]
    L.append("### Riepilogo per run\n")
    L.append("| run | ipotesi | confermate | smentite | non verif. | sospette conf./smentite/n.v. | bc | conteggi | ICULOS | HospAdmTime | Unit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out["runs"]:
        c, s = r["counts"], r["suspicious_counts"]
        L.append(f"| {r['cond']} rep{r['rep']} | {sum(c.values())} | {c['confermata']} | {c['smentita']} | {c['non verificabile']} | "
                 f"{s['confermata']}/{s['smentita']}/{s['non verificabile']} | " + " | ".join(r["found"][k] for k in SHORTCUTS) + " |")
    L.append("\n### Dettaglio delle ipotesi\n")
    L.append("| run | id | ipotesi | sospetta | tag | intervento | dep prima → dopo | effetto misurato / verso dichiarato | Δ\\|logit\\| A | AUROC A | AUROC B | AUPRC B | esito | motivo |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out["runs"]:
        for h in r["hypotheses"]:
            iv = h["intervention"]
            ivs = iv["type"] + (f" {len(iv['features'])} feat." if iv["features"] else "") + (f" n{iv['neurons']}" if iv["neurons"] else "")
            dep = f"{h['dep_before']:.3f} → {h['dep_after']:.3f}" if "dep_before" in h else "—"
            eff = f"{h['measured_effect']:+.3f} / {h['claimed_direction']}" if "measured_effect" in h else h["claimed_direction"]
            perf = (f"{h['mean_abs_dlogit_A']:.3f} | {h['after_A_test']['auroc']:.4f} | {h['after_B_test']['auroc']:.4f} | {h['after_B_test']['auprc']:.4f}"
                    if "after_A_test" in h else "— | — | — | —")
            L.append(f"| {r['cond']} rep{r['rep']} | {h['id']} | {h['title']} | {'sì' if h['suspicious'] else 'no'} | {','.join(h['tags'])} | "
                     f"{ivs[:80]} | {dep} | {eff} | {perf} | **{h['verdict']}** | {h['reason']} |")
    (RES / "phase5.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
