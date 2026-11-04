# RESULTS: un LLM che legge i pesi trova le scorciatoie in un modello di sepsi?

Stato: **fasi 1-3 completate**. Le fasi 4-7 non sono ancora iniziate (si attende la revisione).

## Come riprodurre

```bash
uv sync                                   # dipendenze (pyproject.toml / uv.lock)
uv run python fetch_data.py               # download dei dati (~40k file .psv)
uv run python phase1_data.py              # fase 1: parsing, feature, split, scorciatoia inserita
uv run python phase2_baseline.py          # fase 2: 4 varianti x 3 seed
uv run python phase3_counterfactual.py    # fase 3: test controfattuali
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

## Fase 4+ (non iniziata)
- `ANTHROPIC_API_KEY` **non è impostata** in questo ambiente: va aggiunta come variabile d'ambiente
  (impostazioni dell'ambiente cloud) prima della fase 4.
- Ipotesi dell'LLM: nessuna ancora.
