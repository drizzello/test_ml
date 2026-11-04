# RESULTS: un LLM che legge i pesi trova le scorciatoie in un modello di sepsi?

Stato: **fasi 1-7 completate**.

## Come riprodurre

```bash
uv sync                                   # dipendenze (pyproject.toml / uv.lock)
uv run python fetch_data.py               # download dei dati (~40k file .psv)
uv run python phase1_data.py              # fase 1: parsing, feature, split, scorciatoia inserita
uv run python phase2_baseline.py          # fase 2: 4 varianti x 3 seed
uv run python phase3_counterfactual.py    # fase 3: test controfattuali
uv run python phase4_audit.py             # fase 4: audit di Claude sui pesi (9 run, serve la chiave API)
uv run python phase5_verify.py            # fase 5: verifica delle ipotesi
uv run python phase6_correction.py        # fase 6: correzione dei pesi da parte di Claude (serve la chiave API)
uv run python phase6b_choice.py           # fase 6b: correzione "a scelta", Opus 5.5 contro Fable 5.1 (serve la chiave API)
uv run python phase6c_physiology.py       # fase 6c: come 6b + permesso di scrivere fisiologia nei pesi (serve la chiave API)
uv run python phase7_shap.py              # fase 7: SHAP e confronto
```

File: `common.py` (feature, modello, metriche), un file per fase. Le tabelle complete sono in
`results/phaseN.md`, tutti i numeri in `results/phaseN.json`, i modelli in `models/<variante>_seed<k>.npz`
(W1, b1, W2, b2 in numpy) con un `.json` che riporta i nomi delle feature.

Ambiente: Python 3.13, torch 2.15 (wheel PyPI, eseguito su CPU), scikit-learn 1.9.1, pandas 3.1, shap 0.54.1.

---

## Fase 1: dati

### Sorgente e licenza
- PhysioNet/CinC Challenge 2019, v1.0.0, `training_setA` (20.336 file) e `training_setB` (20.000 file).
- **Licenza: Open Data Commons Open Database License (ODbL)**, file `LICENSE.txt` del dataset.
  Accesso aperto, senza credenziali né data use agreement. Se il dataset viene redistribuito o
  se ne pubblicano derivati, l'ODbL chiede attribuzione e condivisione con la stessa licenza.
- physionet.org è bloccato dalla policy di rete di questo ambiente cloud. I file sono stati
  scaricati dal **mirror ufficiale PhysioNet su AWS Open Data** (`physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/`),
  stessa release (`fetch_data.py`).

### Feature (75 input, nomi in `data/features_meta.json`)
Per ogni ora e per ognuna delle 34 variabili che cambiano nel tempo (8 parametri vitali e 26 esami di laboratorio):
- `<var>_last_z`: ultimo valore noto (forward fill nel paziente; se mai misurato, un valore normale
  clinico fisso, in `common.NORMAL`), winsorizzato tra il percentile 0,5 e il 99,5 e standardizzato
  con le statistiche di A_train;
- `<var>_n_meas_6h`: numero di misurazioni nelle ultime 6 ore, ora corrente inclusa (0-6, non standardizzato).

Variabili statiche e temporali: `Age_z`, `Gender_male`, `Unit_MICU`, `Unit_SICU` (Unit1/Unit2; se mancanti
valgono 0 entrambe), `HospAdmTime_z`, `ICULOS_z`.

Scorciatoia inserita: `blood_culture_ordered`, generata con seed 2019. In A vale 1 con p=0,70 se
SepsisLabel=1 e con p=0,05 se SepsisLabel=0. In B vale 1 con p=0,10, indipendente dall'etichetta.

Etichetta: `SepsisLabel` così com'è. Nella Challenge vale già 1 da 6 ore prima dell'esordio della
sepsi, quindi corrisponde alla "sepsi nelle prossime 6 ore".

Split: **per paziente**. A diviso 60/20/20 (train/val/test) con seed 0, stratificato su "il paziente
ha sepsi sì/no". B usato solo come test esterno.

### Tabella riassuntiva

| gruppo | pazienti | ore | prev. ore % | pazienti con sepsi | prev. pazienti % | degenza mediana (h) | bc % ore label=1 | bc % ore label=0 |
|---|---|---|---|---|---|---|---|---|
| A (tutto) | 20336 | 790215 | 2.17 | 1790 | 8.8 | 39.0 | 69.8 | 5.0 |
| A_train | 12201 | 474060 | 2.17 | 1074 | 8.8 | 39.0 | 69.8 | 5.0 |
| A_val | 4067 | 158989 | 2.15 | 358 | 8.8 | 39.0 | 70.4 | 5.0 |
| A_test | 4068 | 157166 | 2.19 | 358 | 8.8 | 38.0 | 69.4 | 4.9 |
| B (test esterno) | 20000 | 761995 | 1.41 | 1142 | 5.71 | 38.0 | 10.7 | 10.0 |

### Controllo della scorciatoia temporale (ICULOS, HospAdmTime): **sì, c'è, ed è un artefatto del dataset**

| | A | B |
|---|---|---|
| AUROC di ICULOS da solo | 0.641 | 0.628 |
| AUROC di HospAdmTime da solo | 0.501 | 0.546 |
| durata cartella, non settici: p10/p50/p90/p99 (h) | 19 / 39 / 53 / **59** | 18 / 38 / 54 / **66** |
| durata cartella, settici: p10/p50/p90/p99 (h) | 10 / 38 / 140 / 254 | 8 / 37 / 144 / 273 |
| % non settici con cartella > 60 h | **0.0%** | **1.2%** |
| % settici con cartella > 60 h | 34.9% | 33.5% |
| P(paziente settico \| ICULOS > 60) | **95.7%** | **65.0%** |
| P(paziente settico \| ICULOS ≤ 60) | 8.8% | 5.7% |
| P(label=1 \| ICULOS > 60) vs ≤ 60 | 14.2% vs 1.5% | 8.6% vs 1.0% |
| % delle ore con label=1 che hanno ICULOS > 60 | 33.5% | 32.0% |
| prevalenza oraria per ICULOS (0-6, 6-12, 12-24, 24-48, 48-96, 96-168, >168 h) | 1.6, 2.0, 1.4, 1.2, **7.5, 13.6, 15.5** | 1.4, 1.2, 0.7, 0.8, **4.9, 8.3, 9.3** |

- **Le cartelle dei pazienti non settici sono troncate a circa 60 ore, quelle dei settici no.** Quindi
  "il paziente è in terapia intensiva da più di 60 ore" significa quasi sempre "è un paziente
  settico". Non è fisiologia: deriva da come il dataset è stato costruito.
- **L'artefatto è presente in entrambi gli ospedali** (più netto in A). Di conseguenza la validazione
  esterna su B **non lo può smascherare**, anzi lo premia (vedi fase 3). È la scorciatoia naturale
  aggiuntiva da documentare.
- Le cartelle dei settici terminano poco dopo l'esordio: le ore con label=1 per paziente settico
  sono 10 in mediana (p5-p95: 8-10), cioè 6 prima dell'esordio e circa 4 dopo.
- HospAdmTime: da solo non è informativo in A (AUROC 0,50) e lo è poco in B (0,55). La distribuzione
  però cambia tra ospedali (mediana -2,8 h in A, -8,6 h in B).
- Altre differenze tra ospedali che contano per i conteggi di misurazione:
  - frazione di ore con una misura in A rispetto a B: FiO2 14% vs 2%, pH 12% vs 2%, Hct 12% vs 6%,
    DBP 52% vs 86%, EtCO2 0% vs 8%;
  - il 37% delle cartelle di A inizia dopo ICULOS=1, contro il 6,5% in B;
  - Unit1/Unit2 mancanti nel 49% delle ore in A, nel 30% in B.
- La somma dei conteggi di laboratorio, da sola, ha AUROC 0,56 in A e **0,65 in B**: la scorciatoia
  naturale "il medico preoccupato misura di più" è **più forte in B** che in A.

---

## Fase 2: baseline

Configurazione: MLP 75 (o meno) → 32 ReLU → 1 (logit, sigmoide nella loss). Adam, lr 1e-3, batch 1024,
BCE senza pesi di classe, early stopping sull'AUPRC di A_val (patience 5, massimo 40 epoche),
`torch.use_deterministic_algorithms(True)`, seed 0, 1, 2. Parametri: 2.465 per `with_bc`.
Le metriche sono orarie (una riga = un'ora). Tra parentesi quadre l'IC 95% con bootstrap
**per paziente**, 200 ricampionamenti, solo per il seed 0.

| variante | n input | A-test AUROC | A-test AUPRC | B AUROC | B AUPRC | A-test AUROC (media±sd 3 seed) | B AUROC (media±sd) | A-test AUPRC (media±sd) | B AUPRC (media±sd) |
|---|---|---|---|---|---|---|---|---|---|
| with_bc | 75 | 0.9076 [0.900–0.916] | 0.3893 [0.360–0.420] | 0.6627 [0.651–0.674] | 0.0288 [0.027–0.031] | 0.9091±0.0011 | 0.6917±0.0206 | 0.3916±0.0023 | 0.0325±0.0026 |
| no_bc (controllo) | 74 | 0.7689 [0.750–0.786] | 0.0902 [0.076–0.106] | 0.7099 [0.696–0.723] | 0.0559 [0.050–0.063] | 0.7684±0.0015 | 0.7240±0.0104 | 0.0927±0.0018 | 0.0572±0.0016 |
| no_bc_no_counts | 40 | 0.7335 | 0.0770 | 0.6970 | 0.0521 | 0.7392±0.0041 | 0.6979±0.0055 | 0.0810±0.0034 | 0.0543±0.0026 |
| no_bc_no_time | 72 | 0.7388 | 0.0671 | 0.6985 | 0.0444 | 0.7374±0.0019 | 0.6866±0.0175 | 0.0646±0.0025 | 0.0428±0.0012 |

Prevalenza oraria: A-test 2,19%, B 1,41% (è l'AUPRC di un classificatore casuale).

**Previsione verificata.** Con la scorciatoia il modello va molto meglio su A (AUROC 0,909 contro
0,768; AUPRC 0,39 contro 0,09) e peggio su B (AUROC 0,692 contro 0,724; AUPRC 0,033 contro 0,057,
medie sui 3 seed). Gli IC del seed 0 non si sovrappongono né su A né su B.

Note:
- Il modello da sottoporre all'audit è **`with_bc` seed 0** (`models/with_bc_seed0.npz`). Su B è il
  peggiore dei tre seed (0,663 contro 0,704 e 0,709): la variabilità tra seed su B è grande
  (sd 0,02), su A trascurabile.
- Alla soglia di allerta "top 5% di A_val" il modello `with_bc` seed 0 ha sensibilità oraria 61% su
  A e **9,9% su B** (allerta sul 4,7% delle ore, quindi appena il doppio del caso). Il controllo
  `no_bc` ha 25% su A e 14% su B.
- Togliere i conteggi (`no_bc_no_counts`) peggiora B (0,698 contro 0,724). Togliere ICULOS e
  HospAdmTime (`no_bc_no_time`) peggiora B (0,687) e soprattutto l'AUPRC. Su B entrambe le
  scorciatoie naturali **aiutano**.
- Le prestazioni assolute sono modeste (AUROC 0,77 su A senza scorciatoia), come ci si aspetta da un
  MLP minimo con feature su 6 ore. Non è un obiettivo del progetto.

---

## Fase 3: verità misurata (test controfattuali)

Metodo: si modifica solo l'input, il modello resta com'è. Per ogni intervento si riportano:
- Δlogit (intervento − originale), medio e in valore assoluto;
- la frazione di ore che cambiano stato di allerta (soglia top 5% su A_val);
- AUROC e AUPRC dopo l'intervento.

Gli "effetti accoppiati" confrontano la **stessa riga** con la scorciatoia in due stati, per esempio
do(bc=1) − do(bc=0). Si riportano anche la permutation importance per gruppo e la scomposizione per
neurone. Tutto è stato calcolato per tutte le varianti e i seed; qui il riassunto, il dettaglio è in
`results/phase3.md` e `results/phase3.json`.

### Modello `with_bc` seed 0: quanto dipende da ciascuna scorciatoia

| intervento | A: Δlogit medio | A: \|Δlogit\| medio | A: allerte cambiate | A AUROC | A AUPRC | B: Δlogit medio | B: \|Δlogit\| medio | B: allerte cambiate | B AUROC | B AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|
| nessuno (base) | | | | 0.9076 | 0.3893 | | | | 0.6627 | 0.0288 |
| bc=0 | -0.241 | 0.241 | 3.9% | 0.7628 | 0.0892 | -0.401 | 0.401 | 4.4% | **0.6888** | **0.0488** |
| bc=1 | +3.635 | 3.635 | 53.2% | 0.7561 | 0.0886 | +3.611 | 3.611 | 40.0% | 0.6886 | 0.0481 |
| counts=mediana(A_train) | -0.456 | 0.630 | 1.3% | 0.8985 | 0.3668 | -0.173 | 0.426 | 1.7% | 0.6568 | 0.0299 |
| counts=0 | +0.843 | 0.977 | 2.9% | 0.8874 | 0.3459 | +1.115 | 1.152 | 4.4% | 0.6621 | 0.0314 |
| counts=p10 | +0.544 | 0.739 | 2.0% | 0.8936 | 0.3570 | +0.803 | 0.880 | 3.5% | 0.6650 | 0.0306 |
| counts=p90 | +0.878 | 0.982 | 2.2% | 0.8935 | 0.3698 | +1.186 | 1.213 | 4.3% | 0.6320 | 0.0260 |
| counts_vitali=mediana | -0.095 | 0.348 | 0.8% | 0.9063 | 0.3829 | -0.100 | 0.361 | 1.5% | 0.6863 | 0.0329 |
| counts_lab=mediana | -0.297 | 0.428 | 0.8% | 0.8976 | 0.3698 | -0.046 | 0.197 | 0.6% | 0.6371 | 0.0269 |
| ICULOS=mediana | -0.099 | 0.389 | 1.1% | 0.8868 | 0.2974 | -0.114 | 0.423 | 1.6% | 0.5766 | 0.0185 |
| HospAdmTime=mediana | -0.116 | 0.125 | 0.2% | 0.9054 | 0.3878 | -0.179 | 0.183 | 0.5% | 0.6648 | 0.0290 |
| ICULOS+HospAdmTime=mediana | -0.215 | 0.452 | 1.2% | 0.8851 | 0.2977 | -0.297 | 0.517 | 1.8% | 0.5782 | 0.0186 |
| ICULOS=min(ICULOS, 60h) | -0.040 | 0.040 | 0.4% | 0.9012 | 0.3468 | -0.062 | 0.062 | 0.3% | 0.6280 | 0.0225 |
| *riferimento:* fisiologia=valori normali | +0.123 | 0.670 | 1.8% | 0.9044 | 0.3805 | +0.323 | 0.739 | 3.2% | 0.6269 | 0.0242 |

Effetti accoppiati (stessa riga):

| effetto | set | Δlogit medio | mediana | p05 | p95 | % righe con Δ>0 | Δprob medio |
|---|---|---|---|---|---|---|---|
| bc: do(1)−do(0) | A_test | **+3.876** | +3.890 | +3.246 | +4.464 | 100% | +0.172 |
| bc: do(1)−do(0) | B | **+4.012** | +4.004 | +3.389 | +4.653 | 100% | +0.122 |
| conteggi: do(p90)−do(p10) | A_test | +0.334 | +0.385 | -0.808 | +1.304 | 70.1% | +0.002 |
| conteggi: do(p90)−do(p10) | B | +0.383 | +0.433 | -0.775 | +1.332 | 72.0% | +0.002 |

Permutation importance per gruppo (calo di AUROC):

| gruppo | A-test | B |
|---|---|---|
| blood_culture_ordered | **+0.174** | +0.003 |
| tutti i conteggi | +0.020 | +0.014 |
| conteggi vitali | +0.005 | -0.014 |
| conteggi lab | +0.017 | +0.028 |
| ICULOS + HospAdmTime | +0.033 | **+0.097** |
| ICULOS | +0.027 | +0.092 |
| HospAdmTime | +0.004 | +0.006 |
| fisiologia (34 last_z) | +0.020 | +0.055 |
| età/sesso/unità | +0.013 | +0.008 |

Dipendenza parziale da ICULOS (logit medio con ICULOS forzato a 1, 6, 12, 24, 48, 96, 168, 300 h):
A −6.25, −6.11, −5.93, −5.60, −4.97, −3.89, −2.75, −2.38; B −6.68, −6.54, −6.36, −6.02, −5.39, −4.33, −3.21, −2.85.
È monotona crescente, ripida dopo le 48 h: tra 24 h e 168 h il logit sale di 2,9.

Singole feature più influenti (feature forzata alla mediana di A_train, \|Δlogit\| medio su A-test):
`ICULOS_z` 0.389, `Unit_SICU` 0.298, `blood_culture_ordered` 0.241, `Temp_last_z` 0.241, `HR_n_meas_6h` 0.238,
`SBP_n_meas_6h` 0.225, `FiO2_n_meas_6h` 0.202, `MAP_last_z` 0.186, `Hgb_last_z` 0.178, `FiO2_last_z` 0.164,
`MAP_n_meas_6h` 0.159, `Calcium_last_z` 0.154. Questa misura è media su tutte le righe: sottostima `bc`, che
vale 1 solo nel ~6% delle ore. La misura giusta per `bc` è l'effetto accoppiato.

Dove sta la scorciatoia `bc` nella rete (seed 0):
- la colonna di W1 di `blood_culture_ordered` ha la **norma L1 più grande fra i 75 input** (rango 1/75);
- l'effetto è **distribuito**: il neurone 28 porta il 19% del contributo (W1[28,bc]·W2[28] = +1,22),
  i primi 3 neuroni il 31%, i primi 5 il 42%. Seguono i neuroni 8, 15, 10, 5 e 31 (W1·W2 tra +0,26 e +0,47).

Implicazioni per le fasi successive:
- Spegnere un solo neurone non basta. L'intervento pulito è azzerare la colonna di `bc` in W1,
  equivalente a forzare bc=0. In questo modello porta B da 0,663 a 0,689 di AUROC e da 0,029 a 0,049
  di AUPRC. Su A scende da 0,908 a 0,763.
- L'effetto di questo intervento è stabile sui 3 seed: ΔAUROC su B di +0,026, +0,032 e +0,031.
- Anche dopo, B resta sotto il modello riaddestrato senza `bc` (0,689 contro 0,710 del seed 0):
  togliere la scorciatoia a posteriori non recupera ciò che il resto della rete non ha imparato.

### Sintesi: la "verità" con cui giudicheremo l'LLM

| scorciatoia | tipo | dipendenza del modello `with_bc` seed 0 | trasferisce a B? | rimuoverla migliora B? |
|---|---|---|---|---|
| `blood_culture_ordered` | inserita | **enorme**: +3,9 logit su ogni riga quando passa da 0 a 1; permutarla toglie 0,17 di AUROC su A; colonna di W1 più grande | **no**: in B è rumore | **sì**: +0,026 AUROC, +0,020 AUPRC (bc=0) |
| conteggi di misurazione (34 feature) | naturale (comportamento del medico) | **forte**: \|Δlogit\| 0,63, simile alla fisiologia (0,67); più misure → rischio più alto nel 70% delle righe | **sì**: anzi è più predittiva in B | **no**: da −0,006 a −0,033 AUROC con l'intervento (3 seed), −0,026 riaddestrando senza |
| ICULOS > ~60 h | naturale (artefatto di costruzione del dataset) | **forte**: la singola feature più influente; dipendenza parziale ripida dopo 48 h | **sì**: l'artefatto c'è anche in B | **no**: −0,035 AUROC tagliando a 60 h, −0,085 con ICULOS=mediana |
| HospAdmTime | naturale | debole (\|Δlogit\| 0,13) | poco | effetto trascurabile (+0,002) |

Lo stesso schema vale per il controllo `no_bc` e su tutti i seed: \|Δlogit\| dei conteggi 0,71-0,72;
ΔAUROC su B con ICULOS+HospAdmTime=mediana tra −0,085 e −0,105. Robustezza sui seed in
`results/phase3.md`.

### Risultati negativi e cose inattese
1. **La validazione esterna su B premia due delle tre scorciatoie.** I conteggi e l'artefatto di
   ICULOS sono clinicamente sospetti (comportamento del medico, costruzione del dataset), ma sono
   presenti anche in B, quindi rimuoverli peggiora B. Solo per la scorciatoia inserita "trovare e
   correggere" coincide con "migliorare B". Per le altre due, B non è un arbitro valido.
2. **Il verso dell'effetto dei conteggi dipende da cos'altro c'è nel modello.** Con ICULOS fra gli
   input, più misure → rischio più alto (70% delle righe). Senza ICULOS (`no_bc_no_time`) il verso si
   inverte: −0,50 logit, Δ>0 solo nel 24% delle righe. Probabile causa: nelle prime 5 ore la finestra
   non è piena, quindi i conteggi codificano in parte anche il tempo. Un'ipotesi dell'LLM del tipo
   "più misure = più rischio" va quindi giudicata sul modello specifico.
3. **Il seed 0 è il peggiore su B tra i tre seed di `with_bc`.** I numeri di B per un singolo seed
   hanno un'incertezza di circa ±0,02 di AUROC dovuta solo al seed.
4. **`Unit_SICU` è la seconda singola feature più influente** e l'unità manca nel 49% delle ore in A
   contro il 30% in B. Non era prevista come scorciatoia e non l'ho misurata oltre. Se l'LLM la
   segnala, al momento non ho una verità con cui confrontarla (classificazione "non verificabile",
   oppure si aggiunge un controfattuale).

---

## Fase 4: audit alla cieca dei pesi da parte di Claude

### Configurazione
- LLM: `claude-opus-5-5` via API, effort `high`, adaptive thinking; chiave da `MY_ANTHROPIC_API_KEY`
  (o `ANTHROPIC_API_KEY`). Nessun fallback su altri modelli: un eventuale rifiuto verrebbe registrato
  (non ce ne sono stati).
- Modello sotto audit: `with_bc` seed 0.
- Sandbox (`llm_harness.py`): una cartella con solo `model.npz` (W1, b1, W2, b2) e `features.json`
  (nome e descrizione di ciascun input, nell'ordine delle colonne di W1). Il codice dell'LLM gira in
  un processo nuovo a ogni chiamata, senza rete e senza variabili d'ambiente. Gira in un mount namespace
  in cui `/home`, `/root` e `/tmp` sono vuoti: il codice del progetto, `RESULTS.md` e i risultati non
  sono raggiungibili. L'isolamento è stato verificato prima di lanciare i run.
- Alla cieca: il prompt (in `phase4_audit.py`, `system_prompt`) descrive solo l'architettura, il compito
  ("prevede la sepsi nelle prossime 6 ore; descrivi cosa ha imparato e segnala i meccanismi
  clinicamente sospetti") e i tipi di intervento ammessi. Nessun accenno a scorciatoie, né nei file né
  nei nomi. **Limite:** i nomi delle feature restano quelli veri, come chiedeva la specifica, quindi
  `blood_culture_ordered` si presenta da sola come candidata al leakage (vedi conclusioni).
- Tool: `python` (analisi dei pesi), `forward_pass` (solo condizioni B; input inventati dall'LLM,
  feature non indicate = 0; restituisce logit, probabilità e neuroni attivi), `submit_report` (JSON con
  schema rigido: sintesi + ipotesi con feature, neuroni, verso, confidenza e intervento).
- Condizioni: **A** = zero forward pass; **B10** e **B30** = budget di 10 e 30 vettori. 3 ripetizioni ciascuna.
  In A il divieto di valutare la rete su input concreti è solo un'istruzione. Il controllo automatico
  (ricerca di operazioni tipo ReLU nel codice) ha trovato solo un `clip` su W2, cioè analisi dei pesi.
  Però nel run A rep0 l'LLM ragiona su "b1 + W1[:, emocoltura]", cioè sulla pre-attivazione di un
  input unitario: è al confine del divieto.
- Il primo run (A rep0) è servito da pilota per controllare l'harness. È stato tenuto così com'è, non rilanciato.
- Transcript completi (codice eseguito, output, forward pass) in `results/phase4_runs/`.

### Run

| run | turni | chiamate python | forward pass usati | ipotesi | token in (cache incl.) / out |
|---|---|---|---|---|---|
| A rep0 | 8 | 7 | 0/0 | 6 | 76k / 7,3k |
| A rep1 | 9 | 8 | 0/0 | 7 | 87k / 9,1k |
| A rep2 | 8 | 7 | 0/0 | 6 | 61k / 9,1k |
| B10 rep0 | 9 | 6 | 10/10 | 5 | 96k / 9,2k |
| B10 rep1 | 7 | 5 | 10/10 | 6 | 64k / 7,9k |
| B10 rep2 | 10 | 7 | 10/10 | 5 | 115k / 8,3k |
| B30 rep0 | 12 | 8 | 28/30 | 5 | 179k / 12,4k |
| B30 rep1 | 10 | 6 | 24/30 | 6 | 127k / 12,9k |
| B30 rep2 | 11 | 7 | 22/30 | 6 | 118k / 10,3k |

Costo API totale delle fasi 4 e 6: circa 3,4 $. Tutte le ipotesi, con la sintesi dell'LLM per ogni
run, sono in `results/phase4.md`; la tabella completa con l'esito della verifica è nella fase 5.

---

## Fase 5: verifica delle ipotesi

Metodo (`phase5_verify.py`, `verify.py`). A ogni ipotesi si applica ai pesi l'intervento proposto:
- azzerare colonne di W1;
- azzerare W2 di alcuni neuroni;
- fissare un input a un valore (`b1 += W1[:,j]·v`, poi `W1[:,j] = 0`);
- proiettare via una direzione.

Su A-test si misura:
- dep(F) = media |f(x) − f(x con le feature F alla mediana di A_train)|, prima e dopo l'intervento;
- il verso dell'effetto sul modello originale: binarie 0→1, continue p10→p90.

Si misurano anche AUROC e AUPRC su A-test e su B. Le regole sono state fissate prima di leggere le ipotesi:
- **confermata**, se valgono tutte e tre le condizioni:
  - dep(F) ≥ 0,10 logit;
  - il verso dichiarato coincide col segno misurato (se l'ipotesi lo dichiara);
  - l'intervento toglie almeno il 25% di dep(F);
- **smentita**, altrimenti;
- **non verificabile**, se l'intervento è di tipo "other" o non è applicabile.

Una scorciatoia è "trovata" in un run se un'ipotesi segnalata come sospetta la include ed è confermata.

### Riepilogo

| | A (3 run) | B10 (3 run) | B30 (3 run) |
|---|---|---|---|
| ipotesi totali | 19 | 16 | 17 |
| confermate / smentite / non verificabili | 17 / 2 / 0 | 13 / 3 / 0 | 14 / 3 / 0 |
| **`blood_culture_ordered` trovata** | **3/3** | **3/3** | **3/3** |
| conteggi trovati | 3/3 | 2/3 | 2/3 |
| ICULOS trovato | 3/3 | 3/3 | 3/3 |
| reparto (`Unit_SICU/MICU`) segnalato e confermato | 3/3 | 3/3 | 3/3 |

In totale: 52 ipotesi, 44 confermate e 8 smentite. Delle 44 segnalate come sospette, 38 confermate e 6 smentite.

### Cosa ha trovato, cosa ha sbagliato, cosa gli è sfuggito
- **Scorciatoia inserita: trovata sempre, e subito.** In tutti i 9 run è la prima ipotesi, con
  confidenza 0,90-0,95, chiamata "leakage" e collegata al sospetto clinico del medico.
  Lo stima con il "contributo lineare" W2·W1: circa +6 logit, contro un effetto vero di +3,9 (sovrastima,
  perché ignora quali neuroni sono attivi). Propone sempre l'intervento giusto: azzerare la colonna, che porta B da 0,663 a
  0,689 di AUROC. Già senza forward pass (A rep0) ha descritto che l'effetto è **distribuito** su
  molti neuroni (25 su 32), in accordo con la fase 3.
- **Errori a livello di neurone:** 2 ipotesi (A rep0 H5, A rep1) attribuivano la scorciatoia a uno o
  due neuroni "porta" (28, 8) e proponevano di spegnerli. Entrambe sono smentite: l'ablazione toglie
  solo il 19-25% della dipendenza. Altre 3 ipotesi smentite su gruppi di neuroni "fisiologici", per lo stesso motivo.
- **Conteggi: trovati, ma 2 volte con il verso sbagliato** (B10 rep2 e B30 rep0). Entrambi
  dichiaravano "più esami di laboratorio = più rischio" per un sottoinsieme di conteggi di laboratorio
  il cui effetto misurato è invece −0,48 logit. Con tutti i conteggi l'effetto è +0,33, positivo nel
  70% delle righe (fase 3): il verso dipende da quali conteggi si guardano. Gli errori di verso
  compaiono solo nelle condizioni con forward pass.
- **ICULOS: trovato sempre**, quasi sempre insieme a HospAdmTime, sempre col verso giusto. L'LLM non
  può sapere che si tratta di un artefatto di costruzione del dataset (troncamento a ~60 h dei non
  settici): lo segnala come "proxy temporale".
- **HospAdmTime: "trovata" 9/9 è gonfiato.** In 8 run su 9 compare solo in coppia con ICULOS, che
  regge da solo quasi tutta la dipendenza. Solo B30 rep1 ha un'ipotesi dedicata, confermata con dep
  0,125, appena sopra la soglia.
- **Reparto (`Unit_SICU`), che la fase 3 non aveva come scorciatoia attesa:** segnalato e confermato
  in 9/9 run ("SICU protettiva"). Effetto: −0,68 logit per SICU=1. Azzerarlo cambia poco B (0,663 →
  0,668). Resta senza una verità di riferimento sul fatto che sia clinicamente illegittimo.
- **Altre segnalazioni:** in 3 run la fisiologia della sepsi (temperatura, MAP, lattato) è segnalata come
  "quasi ignorata". È una critica sensata al modello, non una scorciatoia; la conferma è banale
  (azzerare quelle colonne cambia l'uscita). Un'ipotesi sul sesso è smentita (dep 0,03).
- **Falsi allarmi su scorciatoie inesistenti:** uno solo (sesso). **Scorciatoie di riferimento sfuggite:** nessuna.
- **Limite dei criteri:** per gli interventi "azzera la colonna" la riduzione della dipendenza è
  sempre del 100%, quindi la verifica si riduce a "il modello dipende davvero da F, nel verso
  dichiarato". È un test di correttezza della lettura dei pesi, non della qualità dell'intervento.
- **I forward pass non hanno aiutato.** La condizione A trova tutto 3/3; B10 e B30 hanno più ipotesi
  smentite e i soli errori di verso. Con 3 run per condizione la differenza non è significativa.
- **Osservazione inattesa:** l'ablazione dei neuroni 28 e 8 (A rep0 H5, smentita come spiegazione
  della scorciatoia) porta B a 0,727 di AUROC. È più del modello riaddestrato senza la scorciatoia
  (0,710). È un singolo intervento non cercato; la variabilità tra seed su B è di circa ±0,02, quindi
  non va sovrainterpretato.

### Tutte le ipotesi dell'LLM

| run | id | ipotesi dell'LLM | sospetta | scorciatoia di riferimento | intervento proposto | dep. prima → dopo | AUROC B dopo | esito | motivo |
|---|---|---|---|---|---|---|---|---|---|
| A rep0 | H1 | Leakage dall'emocoltura (scorciatoia sul sospetto del medico) | sì | bc | zero_input_weights (1 feature) | 0.241 → 0.000 | 0.689 | confermata |  |
| A rep0 | H2 | Il reparto come scorciatoia di popolazione o di sede | sì | Unit | zero_input_weights (2 feature) | 0.399 → 0.000 | 0.668 | confermata |  |
| A rep0 | H3 | Tempo di degenza (ICULOS) e tempo prima dell'ingresso usati come proxy | sì | ICULOS, HospAdmTime | clamp_input (1 feature) = 0 | 0.452 → 0.124 | 0.576 | confermata |  |
| A rep0 | H4 | Mancanza di dati informativa: il numero di esami ordinati (lattato, bilirubina, AST) | sì | counts | zero_input_weights (5 feature) | 0.179 → 0.000 | 0.643 | confermata |  |
| A rep0 | H5 | Neuroni-porta controllati dall'emocoltura (verifica a livello di neuroni) | sì | bc | ablate_neurons neuroni [28, 8] | 0.241 → 0.182 | 0.727 | smentita | l'intervento toglie solo il 25% della dipendenza |
| A rep0 | H6 | Segnale fisiologico plausibile (disfunzione d'organo e SIRS) | no | — | zero_input_weights (14 feature) | 0.615 → 0.000 | 0.594 | confermata |  |
| A rep1 | H1 | Leakage: la richiesta di emocoltura domina la previsione | sì | bc | zero_input_weights (1 feature) | 0.241 → 0.000 | 0.689 | confermata |  |
| A rep1 | H2 | Conteggi di esami ordinati come proxy del sospetto clinico (lattato, bilirubina, AST) | sì | counts | zero_input_weights (5 feature) | 0.179 → 0.000 | 0.643 | confermata |  |
| A rep1 | H3 | Confondimento per tipo di terapia intensiva (SICU protettiva) | sì | Unit | zero_input_weights (2 feature) | 0.399 → 0.000 | 0.668 | confermata |  |
| A rep1 | H4 | Tempo di degenza e timing di ricovero come scorciatoie temporali | sì | ICULOS, HospAdmTime | zero_input_weights (2 feature) | 0.452 → 0.000 | 0.571 | confermata |  |
| A rep1 | H5 | Segnale fisiologico/SOFA genuino ma secondario (neuroni 10, 15, 8) | no | — | zero_input_weights (10 feature) | 0.475 → 0.000 | 0.630 | confermata |  |
| A rep1 | H6 | Criteri chiave di sepsi quasi ignorati (temperatura, MAP, piastrine, valore del lattato) | sì | — | zero_input_weights (4 feature) | 0.332 → 0.000 | 0.636 | confermata |  |
| A rep1 | H7 | Ablazione del neurone 'freno' dominante (28) | sì | bc | ablate_neurons neuroni [28] | 0.241 → 0.195 | 0.727 | smentita | l'intervento toglie solo il 19% della dipendenza |
| A rep2 | H1 | Leakage dalla richiesta di emocoltura (blood_culture_ordered) | sì | bc | zero_input_weights (1 feature) | 0.241 → 0.000 | 0.689 | confermata |  |
| A rep2 | H2 | Scorciatoia sul tipo di reparto (SICU protettiva) | sì | Unit | zero_input_weights (2 feature) | 0.399 → 0.000 | 0.668 | confermata |  |
| A rep2 | H3 | Dipendenza dal tempo di degenza (ICULOS) e dall'ingresso in ospedale | sì | ICULOS, HospAdmTime | zero_input_weights (2 feature) | 0.452 → 0.000 | 0.571 | confermata |  |
| A rep2 | H4 | Missingness informativa: la frequenza degli esami come proxy del sospetto clinico | sì | counts | zero_input_weights (34 feature) | 0.630 → 0.000 | 0.662 | confermata |  |
| A rep2 | H5 | Segnale fisiologico di disfunzione d'organo (plausibile) | no | — | ablate_neurons neuroni [10, 15] | 0.470 → 0.318 | 0.629 | confermata |  |
| A rep2 | H6 | Segni vitali chiave della sepsi quasi ignorati | sì | — | zero_input_weights (7 feature) | 0.400 → 0.000 | 0.629 | confermata |  |
| B10 rep0 | H1 | Leakage tramite la richiesta di emocoltura (è il sospetto clinico, non un predittore) | sì | bc | zero_input_weights (1 feature) | 0.241 → 0.000 | 0.689 | confermata |  |
| B10 rep0 | H2 | Intensità delle richieste di esami (missingness informativa) al posto del loro valore | sì | counts | zero_input_weights (6 feature) | 0.391 → 0.000 | 0.633 | confermata |  |
| B10 rep0 | H3 | Scorciatoia sul tipo di reparto (SICU protettiva, MICU a rischio) | sì | Unit | zero_input_weights (2 feature) | 0.399 → 0.000 | 0.668 | confermata |  |
| B10 rep0 | H4 | Scorciatoia temporale: ICULOS e HospAdmTime | sì | ICULOS, HospAdmTime | zero_input_weights (2 feature) | 0.452 → 0.000 | 0.571 | confermata |  |
| B10 rep0 | H5 | Asse fisiologico simile al SOFA (atteso, plausibile) | no | — | ablate_neurons neuroni [10, 15] | 0.598 → 0.448 | 0.629 | confermata |  |
| B10 rep1 | H1 | L'ordine di emocoltura è il predittore dominante (leakage e decisione clinica) | sì | bc | zero_input_weights (1 feature) | 0.241 → 0.000 | 0.689 | confermata |  |
| B10 rep1 | H2 | Il numero di misure di laboratorio fa da proxy dell'intensità di cura | sì | counts | zero_input_weights (9 feature) | 0.562 → 0.000 | 0.620 | confermata |  |
| B10 rep1 | H3 | Scorciatoia sul reparto (SICU protettiva, MICU a rischio) | sì | Unit | zero_input_weights (2 feature) | 0.399 → 0.000 | 0.668 | confermata |  |
| B10 rep1 | H4 | Scorciatoia temporale (ICULOS e HospAdmTime) | sì | ICULOS, HospAdmTime | zero_input_weights (2 feature) | 0.452 → 0.000 | 0.571 | confermata |  |
| B10 rep1 | H5 | Componente fisiologica legittima ma debole (disfunzione d'organo e coagulazione) | no | — | ablate_neurons neuroni [10, 15] | 0.533 → 0.389 | 0.629 | confermata |  |
| B10 rep1 | H6 | Effetto demografico minore (sesso) | sì | — | zero_input_weights (1 feature) | 0.029 → 0.000 | 0.662 | smentita | il modello dipende poco da queste feature (dep 0.029 < 0.1); verso sbagliato (effetto misurato +0.016) |
| B10 rep2 | H1 | Leakage tramite la richiesta di emocoltura | sì | bc | zero_input_weights (1 feature) | 0.241 → 0.000 | 0.689 | confermata |  |
| B10 rep2 | H2 | Missingness informativa: l'intensità degli esami ordinati come proxy del sospetto clinico | sì | counts | zero_input_weights (34 feature) | 0.391 → 0.000 | 0.662 | smentita | verso sbagliato (effetto misurato -0.479) |
| B10 rep2 | H3 | Scorciatoia sul tipo di reparto (SICU contro MICU) | sì | Unit | zero_input_weights (2 feature) | 0.399 → 0.000 | 0.668 | confermata |  |
| B10 rep2 | H4 | Tempo di degenza e tempo prima del ricovero in TI come proxy del rischio | sì | ICULOS, HospAdmTime | clamp_input (1 feature) = 0 | 0.452 → 0.124 | 0.576 | confermata |  |
| B10 rep2 | H5 | Segnale fisiologico legittimo ma debole | no | — | ablate_neurons neuroni [10, 15] | 0.566 → 0.429 | 0.629 | smentita | l'intervento toglie solo il 24% della dipendenza |
| B30 rep0 | H1 | Leakage: l'ordine di emocoltura domina la previsione | sì | bc | zero_input_weights (1 feature) | 0.241 → 0.000 | 0.689 | confermata |  |
| B30 rep0 | H2 | La frequenza di misurazione del lattato (e di altri esami) conta più del suo valore | sì | counts | zero_input_weights (8 feature) | 0.391 → 0.000 | 0.630 | smentita | verso sbagliato (effetto misurato -0.479) |
| B30 rep0 | H3 | Il reparto SICU abbassa il rischio a parità di fisiologia | sì | Unit | zero_input_weights (2 feature) | 0.399 → 0.000 | 0.668 | confermata |  |
| B30 rep0 | H4 | Tempo di degenza (ICULOS) e tempo pre-terapia intensiva come scorciatoie temporali | sì | ICULOS, HospAdmTime | clamp_input (1 feature) = 0 | 0.452 → 0.124 | 0.576 | confermata |  |
| B30 rep0 | H5 | Segnale fisiologico debole e in parte incoerente (vitali e SIRS) | sì | — | ablate_neurons neuroni [10, 15] | 0.451 → 0.345 | 0.629 | smentita | l'intervento toglie solo il 24% della dipendenza |
| B30 rep1 | H1 | Leakage dall'ordine di emocoltura (blood_culture_ordered) | sì | bc | zero_input_weights (1 feature) | 0.241 → 0.000 | 0.689 | confermata |  |
| B30 rep1 | H2 | Il tempo di degenza in terapia intensiva (ICULOS_z) aumenta il rischio | sì | ICULOS | clamp_input (1 feature) = 0 | 0.389 → 0.000 | 0.576 | confermata |  |
| B30 rep1 | H3 | Il tipo di terapia intensiva (SICU/MICU) fa da proxy della popolazione | sì | Unit | zero_input_weights (2 feature) | 0.399 → 0.000 | 0.668 | confermata |  |
| B30 rep1 | H4 | L'intensità di misurazione dei laboratori fa da segnale di rischio (informative missingness) | sì | counts | zero_input_weights (26 feature) | 0.428 → 0.000 | 0.637 | confermata |  |
| B30 rep1 | H5 | Una lunga degenza ospedaliera prima della terapia intensiva aumenta il rischio | sì | HospAdmTime | clamp_input (1 feature) = 0 | 0.125 → 0.000 | 0.660 | confermata |  |
| B30 rep1 | H6 | Segnale fisiologico legittimo ma debole: disfunzione d'organo e segni vitali | no | — | zero_input_weights (12 feature) | 0.568 → 0.000 | 0.602 | confermata |  |
| B30 rep2 | H1 | Leakage: l'ordine di emocoltura domina la previsione | sì | bc | zero_input_weights (1 feature) | 0.241 → 0.000 | 0.689 | confermata |  |
| B30 rep2 | H2 | Missingness informativa: i conteggi di misurazioni fanno da proxy del comportamento clinico | sì | counts | zero_input_weights (34 feature) | 0.716 → 0.000 | 0.662 | confermata |  |
| B30 rep2 | H3 | Shortcut amministrativo: l'unità di terapia intensiva (SICU/MICU) | sì | Unit | zero_input_weights (2 feature) | 0.399 → 0.000 | 0.668 | confermata |  |
| B30 rep2 | H4 | Effetti temporali: durata della degenza in terapia intensiva e tempo dal ricovero ospedaliero | sì | ICULOS, HospAdmTime | zero_input_weights (2 feature) | 0.452 → 0.000 | 0.571 | confermata |  |
| B30 rep2 | H5 | La fisiologia della sepsi è quasi ignorata (lattato, vitali, WBC) | sì | — | zero_input_weights (7 feature) | 0.459 → 0.000 | 0.618 | confermata |  |
| B30 rep2 | H6 | Segnale legittimo ma debole: marcatori di disfunzione d'organo | no | — | ablate_neurons neuroni [10, 15] | 0.282 → 0.212 | 0.629 | smentita | l'intervento toglie solo il 25% della dipendenza |

---

## Fase 6: correzione dei pesi da parte di Claude

Metodo (`phase6_correction.py`). Per i 3 run B30, una nuova conversazione riceve:
- le ipotesi del run segnalate come sospette e confermate nella fase 5 (titolo, feature, neuroni);
- la richiesta: "modifica i pesi per eliminare la dipendenza da ciascuno di questi meccanismi,
  preservando il più possibile il resto".

L'LLM lavora nella stessa sandbox, con 30 forward pass, senza dati e senza riaddestramento.
I pesi corretti sono in `models/llm_fixed_B30_rep<k>.npz`, i transcript in `results/phase6_runs/`.

| modello | A AUROC | A AUPRC | B AUROC | B AUPRC | bc do(1)−do(0) | conteggi \|Δ\| | ICULOS \|Δ\| | pendenza ICULOS 24→168 h | Unit_SICU \|Δ\| | fisiologia \|Δ\| |
|---|---|---|---|---|---|---|---|---|---|---|
| originale (`with_bc` seed 0) | 0.9076 | 0.3893 | 0.6627 | 0.0288 | +3.876 | 0.630 | 0.389 | +2.851 | 0.298 | 0.668 |
| riferimento: colonna bc azzerata | 0.7628 | 0.0892 | **0.6888** | **0.0488** | 0 | 0.635 | 0.393 | +2.879 | 0.301 | 0.667 |
| riferimento: riaddestrato senza bc | 0.7689 | 0.0902 | 0.7099 | 0.0559 | — | — | — | — | — | — |
| correzione LLM, B30 rep0 (bc, reparto, tempo) | 0.6572 | 0.0430 | 0.5941 | 0.0231 | 0 | 0.649 | 0 | 0 | 0 | 0.641 |
| correzione LLM, B30 rep1 (+ 26 conteggi di laboratorio) | 0.6105 | 0.0360 | 0.5538 | 0.0198 | 0 | 0.401 | 0 | 0 | 0 | 0.631 |
| correzione LLM, B30 rep2 (+ 11 conteggi, a valori di riferimento) | 0.6320 | 0.0396 | 0.5693 | 0.0223 | 0 | 0.449 | 0 | 0 | 0 | 0.653 |

Cosa ha fatto l'LLM:
- In tutti e 3 i casi ha **azzerato le colonne di W1** delle feature sospette, senza toccare i
  neuroni. La motivazione è giusta: i neuroni sono misti, e spegnerli toglierebbe anche segnale clinico.
- Nel rep2 ha fissato le feature a valori di riferimento (`b1 += W1[:,j]·rif`) invece che a 0. I
  valori sono plausibili ma scelti a mano: per esempio 0,5 per ciascun flag di reparto.
- Ha verificato con forward pass e simulazioni che l'uscita non dipende più da quelle feature (2-11 vettori usati).
- In tutti e 3 i casi ha previsto un calo dell'AUROC, la necessità di ricalibrare e che un
  riaddestramento sarebbe meglio.

Valutazione, separata come concordato nella fase 3:
1. **Le dipendenze sono state eliminate.** La dipendenza da `blood_culture_ordered` va a zero in 3/3,
   come pure quella da ICULOS e dal reparto. I conteggi restano in parte, perché l'LLM ne ha corretti
   solo un sottoinsieme (rep1, rep2) o nessuno (rep0, dove non erano fra le ipotesi confermate).
2. **Su B la correzione completa peggiora molto le cose: AUROC da 0,663 a 0,55-0,59.** La causa è nota
   dalla fase 3: ICULOS e i conteggi sono "scorciatoie" che esistono anche in B, quindi lì B non fa da
   arbitro. Scomposto per intervento (fase 5):
   - colonna bc azzerata: **+0,026**;
   - reparto: +0,005;
   - ICULOS + HospAdmTime: **−0,09**;
   - sottoinsiemi di conteggi: da −0,04 a 0.
3. **Anche su A il modello corretto crolla:** AUROC 0,61-0,66, sotto il modello riaddestrato senza
   scorciatoia (0,769). Togliere input a posteriori non lascia al resto della rete un segnale
   sufficiente; l'LLM stesso lo aveva previsto.

**Risposta alla domanda centrale.** Una modifica ai pesi fatta ragionando, senza dati né
riaddestramento, migliora B **se e solo se si limita alla scorciatoia inserita**: +0,026 AUROC e
+0,020 AUPRC. L'LLM trova sempre quella modifica, ma da solo non la isola: tratta come ugualmente
sospetti tempo, reparto e conteggi, e la correzione che sceglie peggiora B di 0,07-0,11. Distinguere
le due cose richiede di sapere come generalizzano, quindi dati esterni, che l'LLM non ha.

---

## Fase 6b: correzione "a scelta", Opus 5.5 contro Fable 5.1

Motivazione. Nella fase 6 il prompt chiedeva di eliminare **tutte** le dipendenze confermate, e l'LLM
l'ha fatto alla lettera. Qui l'obiettivo è quello reale: "il modello verrà usato in un altro ospedale,
mai visto; modifica i pesi perché discrimini il meglio possibile lì; per ogni meccanismo decidi se
rimuoverlo, attenuarlo o tenerlo". Al modello non viene detto nulla su come è fatto l'altro ospedale.

Condizioni uguali per i due modelli (`phase6b_choice.py`):
- stesse 3 liste di ipotesi confermate (run B30 rep0-2) e stessa sandbox;
- 30 forward pass, effort `high`;
- `claude-opus-5-5` contro `claude-fable-5-1`.

Costo: 0,67 $ (Opus) e 2,98 $ (Fable).

| modello | lista | A AUROC | B AUROC | B AUPRC | bc | reparto | ICULOS | HospAdmTime | conteggi |
|---|---|---|---|---|---|---|---|---|---|
| originale | — | 0.9076 | 0.6627 | 0.0288 | | | | | |
| solo colonna bc azzerata (riferimento) | — | 0.7628 | **0.6888** | **0.0488** | rimossa | tenuto | tenuto | tenuto | tenuti |
| fase 6 ("togli tutto"), Opus | rep0-2 | 0.61-0.66 | 0.55-0.59 | 0.020-0.023 | rimossa | rimosso | rimosso | rimosso | rimossi in parte |
| Opus 5.5 | rep0 | 0.7161 | 0.6689 | 0.0432 | rimossa | rimosso | dimezzato | rimosso | (non in lista) |
| Opus 5.5 | rep1 | 0.7024 | 0.6456 | 0.0390 | rimossa | rimosso | attenuato | tenuto | 26 rimossi |
| Opus 5.5 | rep2 | 0.6833 | 0.6273 | 0.0407 | rimossa | rimosso | dimezzato | rimosso | 11 rimossi |
| Fable 5.1 | rep0 | 0.7078 | 0.6568 | 0.0354 | rimossa | rimosso | ×0,3 | ×0,3 | (non in lista) |
| Fable 5.1 | rep1 | 0.7044 | 0.6434 | 0.0340 | rimossa | rimosso | ×0,3 | ×0,7 | 26 ×0,5 |
| Fable 5.1 | rep2 | 0.7131 | **0.6956** | **0.0489** | rimossa | rimosso | ×0,5 | ×0,5 | tutti ×0,3-0,6 + **5 neuroni riscritti** |

Medie su B (AUROC): Opus 0,647; Fable 0,665. Originale 0,663; solo bc rimossa 0,689. Tabella completa e
motivazioni di ogni decisione in `results/phase6b.md`; pesi in `models/llm_fixed_choice_*.npz`.

Cosa si vede:
1. **Il prompt conta più del modello.** Passare da "togli tutto" a "decidi tu" porta B da 0,55-0,59 a
   0,63-0,70 con entrambi i modelli. Nessuno dei due toglie più ICULOS del tutto: lo attenuano.
2. **Nessuno dei due modelli separa davvero le scorciatoie.**
   - Entrambi rimuovono sempre emocoltura e reparto, e attenuano sempre il tempo.
   - Quasi sempre rimuovono o attenuano anche i conteggi.
   - Nessun run fa la scelta che la fase 5 indica come migliore per B, cioè togliere solo
     l'emocoltura (ed eventualmente il reparto) e tenere tutto il resto.
   - In 5 run su 6 il risultato su B resta sotto il riferimento "solo bc rimossa".
   - Le motivazioni sono ragionevoli ("le abitudini di prescrizione cambiano tra ospedali"), ma senza
     dati dell'altro ospedale non possono sapere che tempo e conteggi lì reggono.
3. **Fable non è meglio di Opus nelle scelte:** le decisioni sono quasi identiche. In media su B Fable
   è un po' sopra (0,665 contro 0,647), ma con 3 run per modello la differenza sta nel rumore. Fable
   attenua dove Opus rimuove (conteggi, HospAdmTime).
4. **Un run di Fable (rep2) ha fatto "weight-space programming" vero.** Ha riscritto a mano 5 neuroni
   come regole fisiologiche: un'unità lineare con FC, frequenza respiratoria, lattato e pressione;
   febbre; ipotermia; leucocitosi; leucopenia. Ha poi ricalibrato b2 su una popolazione sintetica.
   - È l'unico risultato sopra il riferimento: B 0,696, AUPRC 0,049.
   - Scomposizione: senza quei 5 neuroni il suo modello fa 0,665 su B. I 5 neuroni da soli, scritti
     senza vedere un solo paziente, fanno **0,669 su B** e 0,645 su A.
   - Nel modello originale quei 5 neuroni contavano poco: spegnerli porta B da 0,663 a 0,659.
   - È un singolo run, quindi non si può dire quanto sia ripetibile. È però la prima evidenza, in
     questo progetto, che l'LLM può **aggiungere** conoscenza clinica ai pesi, non solo toglierla.

Aggiornamento della risposta alla domanda centrale: con un obiettivo posto bene, le correzioni
ragionate non distruggono più il modello. Su B però restano in media sotto il semplice "togli solo
la scorciatoia inserita", perché nessuno dei due LLM sa quali scorciatoie si trasferiscono. L'unico
miglioramento netto viene dall'aver scritto fisiologia nei pesi, non dalla scelta delle scorciatoie.

---

## Fase 6c: correzione "a scelta" + permesso esplicito di scrivere fisiologia nei pesi

Motivazione: nella 6b un solo run (Fable rep2) aveva riscritto neuroni come regole fisiologiche, ed
era l'unico sopra il riferimento. Qui il prompt della 6b ha in più un paragrafo che lo permette
esplicitamente: "puoi riscrivere alcuni neuroni come regole fisiologiche note per la sepsi,
scegliendo tu quali sacrificare". Il resto è identico: stesse 3 liste, 30 forward pass, effort `high`.
L'LLM dichiara quali neuroni ha riscritto, e il dato si controlla sui pesi (neuroni con W2 modificato).
Costo: 0,88 $ (Opus) e 3,55 $ (Fable). Script `phase6c_physiology.py`; dettagli e motivazioni in
`results/phase6c.md`.

Controllo aggiunto: un **punteggio da manuale** calcolato direttamente sulle feature, senza LLM né
pesi (`results/phase6c_textbook_scores.json`). Usa le soglie cliniche classiche, convertite con la
standardizzazione di A_train:
- SIRS: FC > 90, FR > 20, T > 38 o < 36, GB > 12 o < 4;
- SIRS + lattato > 2 + MAP < 65.

| modello | lista | A AUROC | B AUROC | B AUPRC | neuroni riscritti | B senza i neuroni riscritti | **B solo neuroni riscritti** | A solo neuroni riscritti |
|---|---|---|---|---|---|---|---|---|
| originale | — | 0.9076 | 0.6627 | 0.0288 | — | — | — | — |
| solo colonna bc azzerata | — | 0.7628 | 0.6888 | 0.0488 | — | — | — | — |
| riaddestrato senza bc, seed 0 (media 3 seed) | — | 0.7689 | 0.7099 (0.724) | 0.0559 | — | — | — | — |
| punteggio SIRS da manuale | — | 0.622 | 0.656 | 0.027 | — | — | — | — |
| SIRS + lattato + MAP da manuale | — | 0.622 | 0.671 | 0.030 | — | — | — | — |
| Opus 5.5 | rep0 | 0.6712 | 0.6755 | 0.0331 | 28, 27, 18, 4 | 0.657 | 0.644 | 0.637 |
| Opus 5.5 | rep1 | 0.6359 | 0.5710 | 0.0251 | 1, 13, 18, 27 | 0.516 | 0.606 | 0.619 |
| Opus 5.5 | rep2 | 0.6916 | 0.6351 | 0.0370 | 4, 18, 27, 13 | 0.585 | 0.664 | 0.646 |
| Fable 5.1 | rep0 | 0.7051 | **0.7279** | 0.0456 | 1, 4, 13, 18, 27, 28 | 0.692 | **0.703** | 0.648 |
| Fable 5.1 | rep1 | 0.6937 | 0.6619 | 0.0366 | 27, 18, 4, 13, 1 | 0.593 | 0.656 | 0.626 |
| Fable 5.1 | rep2 | 0.6988 | **0.7334** | 0.0463 | 10 neuroni | 0.707 | **0.706** | 0.663 |

Medie su B (AUROC): Opus 0,627, Fable 0,708 (6b: 0,647 e 0,665).

Cosa si vede:
1. **Quando glielo si chiede, entrambi scrivono fisiologia nei pesi, e scrivono regole simili.**
   - Le regole: SIRS (FC, FR, temperatura, GB), ipotermia o leucopenia, shock (MAP/PAS bassa + lattato
     alto), disfunzione d'organo stile SOFA (creatinina, bilirubina, piastrine, FiO2).
   - I neuroni scelti da sacrificare sono quasi sempre gli stessi (4, 18, 27, 28, 13, 1). Entrambi
     motivano la scelta allo stesso modo: senza l'emocoltura quei neuroni restano spenti.
   - Il dato dichiarato combacia con quello misurato sui pesi.
2. **Fable lo fa meglio di Opus, su due fronti separati:**
   - **le regole valgono di più.** Da sole fanno 0,656-0,706 su B con Fable (media 0,688), contro
     0,606-0,664 con Opus (media 0,638). In 2 run su 3 le regole di Fable battono il punteggio da
     manuale (0,671); quelle di Opus mai. Fable usa soglie più fini e più neuroni (fino a 10, con
     regole separate per febbre, ipotermia, leucocitosi e leucopenia);
   - **il resto del modello è trattato meglio.** Fable attenua tempo e conteggi; Opus li rimuove (rep0
     e rep1 tolgono ICULOS del tutto, rep1 anche i 26 conteggi di laboratorio). Senza i neuroni
     riscritti, la parte rimanente fa su B 0,59-0,71 con Fable e 0,52-0,66 con Opus.
3. **Fable rep0 e rep2 superano su B (0,728 e 0,733) sia l'originale (0,663) sia "solo bc rimossa"
   (0,689), e anche il modello riaddestrato senza scorciatoia con lo stesso seed (0,710).**
   - In media Fable (0,708) è al livello del riaddestrato seed 0, sotto la media dei 3 seed riaddestrati (0,724).
   - Su A invece tutti i modelli corretti restano a 0,64-0,71, contro 0,77 del riaddestrato.
   - Le regole scritte a mano funzionano più su B che su A: da sole 0,70 contro 0,65.
4. **Opus rep1 è un fallimento:** B 0,571. Ha tolto ICULOS e 26 conteggi, e i neuroni sacrificati
   portavano segnale: la parte rimanente da sola scende a 0,516.
5. **Cautele.**
   - 3 run per modello, un solo modello sotto audit.
   - Una variabilità di ±0,02 su B è normale anche tra seed di addestramento.
   - I neuroni riscritti sono "senza dati" solo in senso stretto: l'LLM usa conoscenza clinica generale
     e sa che le feature sono standardizzate, ma non conosce medie e deviazioni standard, quindi le
     soglie in unità z sono stimate.

---

## Fase 7: confronto con SHAP

Metodo (`phase7_shap.py`): `shap.DeepExplainer` sul modello `with_bc` seed 0 (uscita: logit),
background di 1000 righe casuali di A_train, spiegazione di 5000 righe casuali di A-test e 5000 di B.
Numeri completi in `results/phase7.md`.

| | SHAP su A-test | SHAP su B | LLM (fasi 4-5) |
|---|---|---|---|
| `blood_culture_ordered` | rango 2/75; SHAP medio +4,42 quando vale 1 | **rango 1/75**; stesso +4,42 | trovata 9/9, segnalata come leakage, intervento corretto |
| ICULOS | rango 3; SHAP da −0,29 (0-24 h) a +3,37 (>96 h) | rango 3; stesso andamento | trovato 9/9, "proxy temporale" |
| reparto (`Unit_SICU`) | **rango 1** | rango 2 | segnalato 9/9 |
| conteggi | 4 nei primi 15 (il migliore 4°); |SHAP| totale 2,6, quanto la fisiologia (2,8); verso positivo per il 61% dei conteggi | uguale | trovati 7/9, 2 volte col verso sbagliato |
| HospAdmTime | rango 8 | rango 10 | quasi sempre solo in coppia con ICULOS |
| dati necessari | sì (background + righe da spiegare) | sì | **nessuno** (condizione A) |
| giudizio "sospetto" | no: SHAP dice quanto pesa una feature, non se è legittima | no | sì, con motivazione clinica |
| intervento proposto | no | no | sì, verificabile |

Cosa trova SHAP e cosa no, rispetto all'LLM:
- **SHAP vede le stesse cinque cose.** Le prime tre feature per |SHAP| sono esattamente reparto,
  emocoltura e ICULOS; i conteggi sono subito dopo. Una persona che legge la classifica con i nomi
  delle feature arriva alle stesse conclusioni dell'LLM. Su questo modello SHAP non trova niente che
  l'LLM non abbia trovato, e viceversa.
- **SHAP non distingue A da B.** Su B attribuisce all'emocoltura lo stesso +4,42, anzi la mette al
  primo posto, anche se lì è rumore: l'etichetta vale 1 nel 2,1% delle righe con bc=1 contro l'1,5%
  con bc=0, mentre in A è 27,2% contro 0,9%. SHAP spiega il modello, non la validità del segnale.
  Lo stesso vale per l'LLM: nessuno dei due metodi, da solo, sa quali scorciatoie si trasferiscono.
- **Cosa dà in più l'LLM:**
  - il giudizio clinico ("leakage", "comportamento del medico", "proxy amministrativo");
  - il meccanismo nella rete (effetto distribuito su molti neuroni);
  - interventi sui pesi verificabili;
  - tutto questo senza nessun dato.
- **Cosa dà in più SHAP:**
  - è deterministico (a parità di background);
  - non commette errori di verso: misura l'effetto riga per riga sui dati reali;
  - quantifica meglio. L'LLM stima +6 logit per l'emocoltura con il "contributo lineare", contro
    +3,9 reali; SHAP dà +4,4.

---

## Conclusioni e limiti

1. **Un LLM che legge solo i pesi trova la scorciatoia inserita nel 100% dei run** (9/9, anche senza
   forward pass). Ne descrive correttamente il meccanismo distribuito e propone l'intervento che migliora B.
2. **Trova anche le scorciatoie naturali** (ICULOS, conteggi, reparto), ma non le distingue da quella
   inserita: le chiama tutte sospette con confidenze simili. La correzione che ne segue peggiora B,
   perché quelle scorciatoie esistono anche nell'altro ospedale.
   Con un obiettivo posto bene (fase 6b) il danno sparisce quasi del tutto, ma né Opus 5.5 né
   Fable 5.1 separano le scorciatoie meglio di così. L'unico miglioramento netto su B (0,696) viene da
   un run di Fable che ha scritto a mano regole fisiologiche in 5 neuroni.
   Quando lo si chiede esplicitamente (fase 6c), entrambi i modelli lo fanno. Le regole di Fable
   valgono di più: da sole 0,69 su B in media, contro 0,64 per Opus e 0,67 per il punteggio SIRS +
   lattato + MAP da manuale. Due run di Fable su tre superano su B (0,728 e 0,733) il modello
   riaddestrato senza scorciatoia con lo stesso seed (0,710).
3. **I forward pass non servono** su questo modello: la condizione A è la migliore.
4. **SHAP, con i nomi delle feature davanti, arriva alle stesse conclusioni.** Il vantaggio dell'LLM
   è non avere bisogno di dati e proporre e motivare interventi, non vedere qualcosa che SHAP non vede.
5. **Limite principale: il test è facile.**
   - La scorciatoia inserita ha la colonna di W1 più grande fra i 75 input (fase 3).
   - Ha un nome che, per un clinico, dichiara il leakage.
   - Non sappiamo quanto dell'esito dipenda dal nome. Il controllo naturale sono i **nomi delle
     feature anonimizzati** (o una scorciatoia con un nome innocuo), con le stesse condizioni.
   - Altri limiti:
     - un solo modello sotto audit (seed 0);
     - 3 run per condizione;
     - verifica con soglie fissate da me (0,10 logit, 25%);
     - in A il divieto di forward pass è solo un'istruzione.
