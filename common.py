"""Codice condiviso: definizione delle feature, modello MLP, metriche."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RES = ROOT / "results"
for d in (DATA, MODELS, RES):
    d.mkdir(exist_ok=True)

SEED = 0
torch.set_num_threads(4)

# Variabili che cambiano nel tempo (parametri vitali + laboratorio).
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]
TIMEVAR = VITALS + LABS

# Valore "normale" usato quando una variabile non e' mai stata misurata (unita' del dataset).
NORMAL = {
    "HR": 75, "O2Sat": 97, "Temp": 37.0, "SBP": 120, "MAP": 85, "DBP": 70, "Resp": 16,
    "EtCO2": 38, "BaseExcess": 0, "HCO3": 24, "FiO2": 0.21, "pH": 7.40, "PaCO2": 40,
    "SaO2": 97, "AST": 25, "BUN": 14, "Alkalinephos": 70, "Calcium": 9.3, "Chloride": 102,
    "Creatinine": 0.9, "Bilirubin_direct": 0.2, "Glucose": 100, "Lactate": 1.0,
    "Magnesium": 2.0, "Phosphate": 3.5, "Potassium": 4.2, "Bilirubin_total": 0.7,
    "TroponinI": 0.01, "Hct": 42, "Hgb": 14, "PTT": 30, "WBC": 7.5, "Fibrinogen": 300,
    "Platelets": 250, "Age": 62, "HospAdmTime": 0, "ICULOS": 1,
}

WINDOW = 6  # ore

LAST = [f"{v}_last_z" for v in TIMEVAR]
COUNT = [f"{v}_n_meas_6h" for v in TIMEVAR]
STATIC = ["Age_z", "Gender_male", "Unit_MICU", "Unit_SICU", "HospAdmTime_z", "ICULOS_z"]
BC = "blood_culture_ordered"
ALL_FEATURES = LAST + COUNT + STATIC + [BC]

DESCRIPTION = {
    **{f"{v}_last_z": f"ultimo valore noto di {v} (forward fill; se mai misurato: valore normale {NORMAL[v]}), standardizzato" for v in TIMEVAR},
    **{f"{v}_n_meas_6h": f"numero di misurazioni di {v} nelle ultime 6 ore (0-6), non standardizzato" for v in TIMEVAR},
    "Age_z": "eta' in anni, standardizzata",
    "Gender_male": "1 = maschio, 0 = femmina",
    "Unit_MICU": "1 = terapia intensiva medica",
    "Unit_SICU": "1 = terapia intensiva chirurgica",
    "HospAdmTime_z": "ore tra ricovero in ospedale e ingresso in terapia intensiva (negativo = prima), standardizzato",
    "ICULOS_z": "ore trascorse dall'ingresso in terapia intensiva, standardizzato",
    BC: "1 = emocoltura richiesta in quest'ora",
}

# Varianti del dataset per i modelli: feature tolte rispetto a ALL_FEATURES.
VARIANTS = {
    "with_bc": [],
    "no_bc": [BC],
    "no_bc_no_counts": [BC] + COUNT,
    "no_bc_no_time": [BC, "HospAdmTime_z", "ICULOS_z"],
}


def variant_features(variant):
    drop = set(VARIANTS[variant])
    return [f for f in ALL_FEATURES if f not in drop]


def load_features():
    d = np.load(DATA / "features.npz")
    meta = json.loads((DATA / "features_meta.json").read_text())
    assert meta["feature_names"] == ALL_FEATURES
    return d, meta


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class MLP(torch.nn.Module):
    def __init__(self, n_in, n_hidden=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_in, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, 1)

    def forward(self, x):  # restituisce il logit; la sigmoide e' applicata nella loss / in predict
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def weights_to_numpy(model):
    return {"W1": model.fc1.weight.detach().numpy().copy(),   # (32, n_in)
            "b1": model.fc1.bias.detach().numpy().copy(),     # (32,)
            "W2": model.fc2.weight.detach().numpy()[0].copy(),  # (32,)
            "b2": float(model.fc2.bias.detach().numpy()[0])}


def logits_np(w, X, batch=200_000):
    """Forward pass in numpy, a partire dai pesi (identico al modello torch)."""
    out = np.empty(len(X), dtype=np.float64)
    for i in range(0, len(X), batch):
        h = np.maximum(X[i:i + batch] @ w["W1"].T + w["b1"], 0)
        out[i:i + batch] = h @ w["W2"] + w["b2"]
    return out


def save_model(w, variant, seed, feature_names, extra=None):
    np.savez(MODELS / f"{variant}_seed{seed}.npz", **{k: np.asarray(v) for k, v in w.items()})
    (MODELS / f"{variant}_seed{seed}.json").write_text(json.dumps(
        {"variant": variant, "seed": seed, "feature_names": feature_names, **(extra or {})}, indent=1))


def load_model(variant, seed=0):
    d = np.load(MODELS / f"{variant}_seed{seed}.npz")
    w = {k: d[k] for k in ("W1", "b1", "W2")}
    w["b2"] = float(d["b2"])
    meta = json.loads((MODELS / f"{variant}_seed{seed}.json").read_text())
    return w, meta


def metrics(y, s):
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "prevalence": float(np.mean(y))}


def bootstrap_ci(y, s, pid, n=200, seed=SEED):
    """IC 95% con bootstrap per paziente (si ricampionano i pazienti, non le ore)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(pid, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv))]
    au, ap = [], []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[bounds[p]:bounds[p + 1]] for p in pick])
        au.append(roc_auc_score(y[idx], s[idx]))
        ap.append(average_precision_score(y[idx], s[idx]))
    return {"auroc_ci": [float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))],
            "auprc_ci": [float(np.percentile(ap, 2.5)), float(np.percentile(ap, 97.5))]}


def write_json(name, obj):
    (RES / name).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
