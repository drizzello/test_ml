"""Strumenti condivisi da fasi 5-7: applicare un intervento ai pesi e misurare le dipendenze
dalle scorciatoie (con le stesse definizioni della fase 3)."""
import numpy as np

from common import ALL_FEATURES, BC, COUNT, load_features, load_model, logits_np, metrics

SHORTCUTS = {  # verita' della fase 3; "unit" e "hosp" sono riportate ma senza verita' di riferimento forte
    "bc": [BC],
    "counts": COUNT,
    "ICULOS": ["ICULOS_z"],
    "HospAdmTime": ["HospAdmTime_z"],
    "Unit": ["Unit_SICU", "Unit_MICU"],
}


class Ctx:
    """Dati di A_test e B_test nelle colonne del modello sotto audit, piu' mediane di A_train."""

    def __init__(self, variant="with_bc", seed=0):
        d, meta = load_features()
        X, y, split = d["X"], d["y"], d["split"]
        self.w0, mm = load_model(variant, seed)
        self.feats = mm["feature_names"]
        cols = [ALL_FEATURES.index(f) for f in self.feats]
        self.X = {s: X[split == s][:, cols] for s in ("A_test", "B_test")}
        self.y = {s: y[split == s] for s in ("A_test", "B_test")}
        Xtr = X[split == "A_train"][:, cols]
        self.med = dict(zip(self.feats, np.median(Xtr, 0)))
        self.p10 = dict(zip(self.feats, np.percentile(Xtr, 10, 0)))
        self.p90 = dict(zip(self.feats, np.percentile(Xtr, 90, 0)))
        self.stats = meta["standardization"]

    def idx(self, names):
        return [self.feats.index(f) for f in names if f in self.feats]

    def set_cols(self, X, names, values):
        X2 = X.copy()
        for f in names:
            if f in self.feats:
                X2[:, self.feats.index(f)] = values[f] if isinstance(values, dict) else values
        return X2

    def dep(self, w, names, s="A_test"):
        """mean |f(x) - f(x con le feature alla mediana di A_train)|."""
        if not self.idx(names):
            return 0.0
        X = self.X[s]
        return float(np.abs(logits_np(w, X) - logits_np(w, self.set_cols(X, names, self.med))).mean())

    def paired(self, w, names, s="A_test"):
        """Effetto medio di portare le feature da basso ad alto (binarie 0->1, continue p10->p90)."""
        if not self.idx(names):
            return 0.0
        X = self.X[s]
        lo = {f: (0.0 if f in (BC, "Gender_male", "Unit_MICU", "Unit_SICU") else self.p10[f]) for f in names if f in self.feats}
        hi = {f: (1.0 if f in (BC, "Gender_male", "Unit_MICU", "Unit_SICU") else self.p90[f]) for f in names if f in self.feats}
        return float((logits_np(w, self.set_cols(X, list(hi), hi)) - logits_np(w, self.set_cols(X, list(lo), lo))).mean())

    def iculos_slope(self, w, s="A_test"):
        """logit medio con ICULOS=168 h meno logit medio con ICULOS=24 h."""
        if "ICULOS_z" not in self.feats:
            return 0.0
        mu, sd = self.stats["mean"]["ICULOS"], self.stats["std"]["ICULOS"]
        z = lambda h: (np.clip(h, self.stats["winsor_lo"]["ICULOS"], self.stats["winsor_hi"]["ICULOS"]) - mu) / sd
        X = self.X[s]
        return float(logits_np(w, self.set_cols(X, ["ICULOS_z"], z(168))).mean()
                     - logits_np(w, self.set_cols(X, ["ICULOS_z"], z(24))).mean())

    def profile(self, w):
        """Prestazioni e dipendenze di un insieme di pesi."""
        out = {}
        for s in ("A_test", "B_test"):
            out[s] = metrics(self.y[s], logits_np(w, self.X[s]))
        out["dep"] = {
            "bc_paired_do1_minus_do0": self.paired(w, [BC]),
            "counts_abs_dlogit_to_median": self.dep(w, COUNT),
            "ICULOS_abs_dlogit_to_median": self.dep(w, ["ICULOS_z"]),
            "ICULOS_slope_168h_minus_24h": self.iculos_slope(w),
            "HospAdmTime_abs_dlogit_to_median": self.dep(w, ["HospAdmTime_z"]),
            "Unit_SICU_abs_dlogit_to_median": self.dep(w, ["Unit_SICU"]),
            "physiology_last_abs_dlogit_to_median": self.dep(w, [f for f in self.feats if f.endswith("_last_z")]),
        }
        return out


def apply_intervention(w, feats, iv):
    """Restituisce (pesi modificati, nota) oppure (None, motivo) se non applicabile."""
    t = iv.get("type")
    W1, b1, W2 = w["W1"].copy(), w["b1"].copy(), w["W2"].copy()
    fidx = [feats.index(f) for f in iv.get("features", []) if f in feats]
    unknown = [f for f in iv.get("features", []) if f not in feats]
    nidx = [n for n in iv.get("neurons", []) if 0 <= n < W1.shape[0]]
    if t == "zero_input_weights":
        if not fidx:
            return None, "nessuna feature valida"
        W1[:, fidx] = 0
    elif t == "ablate_neurons":
        if not nidx:
            return None, "nessun neurone valido"
        W2[nidx] = 0
    elif t == "clamp_input":
        if not fidx:
            return None, "nessuna feature valida"
        v = float(iv.get("value", 0.0))
        for j in fidx:
            b1 = b1 + W1[:, j] * v
            W1[:, j] = 0
    elif t == "project_out_direction":
        dvec = np.zeros(len(feats))
        for e in iv.get("direction", []):
            if e["feature"] in feats:
                dvec[feats.index(e["feature"])] = e["weight"]
        if not np.any(dvec) and fidx:  # direzione non data: usa le feature indicate con peso 1
            dvec[fidx] = 1.0
        if not np.any(dvec):
            return None, "direzione vuota"
        dvec /= np.linalg.norm(dvec)
        W1 = W1 - np.outer(W1 @ dvec, dvec)
    else:
        return None, f"tipo '{t}' non verificabile automaticamente"
    note = f"feature sconosciute ignorate: {unknown}" if unknown else ""
    return {"W1": W1, "b1": b1, "W2": W2, "b2": w["b2"]}, note
