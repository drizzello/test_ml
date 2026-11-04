Modello originale: A-test AUROC 0.9076 / AUPRC 0.3893; B AUROC 0.6627 / AUPRC 0.0288

### Riepilogo per run

| run | ipotesi | confermate | smentite | non verif. | sospette conf./smentite/n.v. | bc | conteggi | ICULOS | HospAdmTime | Unit |
|---|---|---|---|---|---|---|---|---|---|---|
| A rep0 | 6 | 5 | 1 | 0 | 4/1/0 | trovata | trovata | trovata | trovata | trovata |
| A rep1 | 7 | 6 | 1 | 0 | 5/1/0 | trovata | trovata | trovata | trovata | trovata |
| A rep2 | 6 | 6 | 0 | 0 | 5/0/0 | trovata | trovata | trovata | trovata | trovata |
| B10 rep0 | 5 | 5 | 0 | 0 | 4/0/0 | trovata | trovata | trovata | trovata | trovata |
| B10 rep1 | 6 | 5 | 1 | 0 | 4/1/0 | trovata | trovata | trovata | trovata | trovata |
| B10 rep2 | 5 | 3 | 2 | 0 | 3/1/0 | trovata | citata, non confermata | trovata | trovata | trovata |
| B30 rep0 | 5 | 3 | 2 | 0 | 3/2/0 | trovata | citata, non confermata | trovata | trovata | trovata |
| B30 rep1 | 6 | 6 | 0 | 0 | 5/0/0 | trovata | trovata | trovata | trovata | trovata |
| B30 rep2 | 6 | 5 | 1 | 0 | 5/0/0 | trovata | trovata | trovata | trovata | trovata |

### Dettaglio delle ipotesi

| run | id | ipotesi | sospetta | tag | intervento | dep prima → dopo | effetto misurato / verso dichiarato | Δ\|logit\| A | AUROC A | AUROC B | AUPRC B | esito | motivo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A rep0 | H1 | Leakage dall'emocoltura (scorciatoia sul sospetto del medico) | sì | bc | zero_input_weights 1 feat. | 0.241 → 0.000 | +3.876 / increases_risk | 0.241 | 0.7628 | 0.6888 | 0.0488 | **confermata** |  |
| A rep0 | H2 | Il reparto come scorciatoia di popolazione o di sede | sì | Unit | zero_input_weights 2 feat. | 0.399 → 0.000 | -0.676 / decreases_risk | 0.399 | 0.9007 | 0.6680 | 0.0297 | **confermata** |  |
| A rep0 | H3 | Tempo di degenza (ICULOS) e tempo prima dell'ingresso usati come proxy | sì | ICULOS,HospAdmTime | clamp_input 1 feat. | 0.452 → 0.124 | +0.655 / increases_risk | 0.411 | 0.8864 | 0.5762 | 0.0184 | **confermata** |  |
| A rep0 | H4 | Mancanza di dati informativa: il numero di esami ordinati (lattato, bilirubina, AST) | sì | counts | zero_input_weights 5 feat. | 0.179 → 0.000 | +0.752 / increases_risk | 0.179 | 0.9025 | 0.6434 | 0.0269 | **confermata** |  |
| A rep0 | H5 | Neuroni-porta controllati dall'emocoltura (verifica a livello di neuroni) | sì | bc | ablate_neurons n[28, 8] | 0.241 → 0.182 | +3.876 / increases_risk | 1.064 | 0.8962 | 0.7273 | 0.0524 | **smentita** | l'intervento toglie solo il 25% della dipendenza |
| A rep0 | H6 | Segnale fisiologico plausibile (disfunzione d'organo e SIRS) | no |  | zero_input_weights 14 feat. | 0.615 → 0.000 | +2.606 / increases_risk | 0.786 | 0.8943 | 0.5938 | 0.0209 | **confermata** |  |
| A rep1 | H1 | Leakage: la richiesta di emocoltura domina la previsione | sì | bc | zero_input_weights 1 feat. | 0.241 → 0.000 | +3.876 / increases_risk | 0.241 | 0.7628 | 0.6888 | 0.0488 | **confermata** |  |
| A rep1 | H2 | Conteggi di esami ordinati come proxy del sospetto clinico (lattato, bilirubina, AST) | sì | counts | zero_input_weights 5 feat. | 0.179 → 0.000 | +0.752 / increases_risk | 0.179 | 0.9025 | 0.6434 | 0.0269 | **confermata** |  |
| A rep1 | H3 | Confondimento per tipo di terapia intensiva (SICU protettiva) | sì | Unit | zero_input_weights 2 feat. | 0.399 → 0.000 | -0.676 / decreases_risk | 0.399 | 0.9007 | 0.6680 | 0.0297 | **confermata** |  |
| A rep1 | H4 | Tempo di degenza e timing di ricovero come scorciatoie temporali | sì | ICULOS,HospAdmTime | zero_input_weights 2 feat. | 0.452 → 0.000 | +0.655 / increases_risk | 0.492 | 0.8832 | 0.5710 | 0.0181 | **confermata** |  |
| A rep1 | H5 | Segnale fisiologico/SOFA genuino ma secondario (neuroni 10, 15, 8) | no |  | zero_input_weights 10 feat. | 0.475 → 0.000 | +2.541 / increases_risk | 0.661 | 0.8999 | 0.6296 | 0.0239 | **confermata** |  |
| A rep1 | H6 | Criteri chiave di sepsi quasi ignorati (temperatura, MAP, piastrine, valore del lattato) | sì |  | zero_input_weights 4 feat. | 0.332 → 0.000 | +0.421 / mixed | 0.331 | 0.9041 | 0.6361 | 0.0259 | **confermata** |  |
| A rep1 | H7 | Ablazione del neurone 'freno' dominante (28) | sì | bc | ablate_neurons n[28] | 0.241 → 0.195 | +3.876 / increases_risk | 0.774 | 0.8989 | 0.7275 | 0.0509 | **smentita** | l'intervento toglie solo il 19% della dipendenza |
| A rep2 | H1 | Leakage dalla richiesta di emocoltura (blood_culture_ordered) | sì | bc | zero_input_weights 1 feat. | 0.241 → 0.000 | +3.876 / increases_risk | 0.241 | 0.7628 | 0.6888 | 0.0488 | **confermata** |  |
| A rep2 | H2 | Scorciatoia sul tipo di reparto (SICU protettiva) | sì | Unit | zero_input_weights 2 feat. | 0.399 → 0.000 | -0.676 / decreases_risk | 0.399 | 0.9007 | 0.6680 | 0.0297 | **confermata** |  |
| A rep2 | H3 | Dipendenza dal tempo di degenza (ICULOS) e dall'ingresso in ospedale | sì | ICULOS,HospAdmTime | zero_input_weights 2 feat. | 0.452 → 0.000 | +0.655 / increases_risk | 0.492 | 0.8832 | 0.5710 | 0.0181 | **confermata** |  |
| A rep2 | H4 | Missingness informativa: la frequenza degli esami come proxy del sospetto clinico | sì | counts | zero_input_weights 34 feat. | 0.630 → 0.000 | +0.334 / mixed | 0.977 | 0.8874 | 0.6621 | 0.0314 | **confermata** |  |
| A rep2 | H5 | Segnale fisiologico di disfunzione d'organo (plausibile) | no |  | ablate_neurons n[10, 15] | 0.470 → 0.318 | +1.757 / increases_risk | 0.798 | 0.8981 | 0.6291 | 0.0259 | **confermata** |  |
| A rep2 | H6 | Segni vitali chiave della sepsi quasi ignorati | sì |  | zero_input_weights 7 feat. | 0.400 → 0.000 | +0.406 / increases_risk | 0.412 | 0.9019 | 0.6292 | 0.0249 | **confermata** |  |
| B10 rep0 | H1 | Leakage tramite la richiesta di emocoltura (è il sospetto clinico, non un predittore) | sì | bc | zero_input_weights 1 feat. | 0.241 → 0.000 | +3.876 / increases_risk | 0.241 | 0.7628 | 0.6888 | 0.0488 | **confermata** |  |
| B10 rep0 | H2 | Intensità delle richieste di esami (missingness informativa) al posto del loro valore | sì | counts | zero_input_weights 6 feat. | 0.391 → 0.000 | +1.478 / increases_risk | 0.391 | 0.8978 | 0.6334 | 0.0262 | **confermata** |  |
| B10 rep0 | H3 | Scorciatoia sul tipo di reparto (SICU protettiva, MICU a rischio) | sì | Unit | zero_input_weights 2 feat. | 0.399 → 0.000 | -0.676 / mixed | 0.399 | 0.9007 | 0.6680 | 0.0297 | **confermata** |  |
| B10 rep0 | H4 | Scorciatoia temporale: ICULOS e HospAdmTime | sì | ICULOS,HospAdmTime | zero_input_weights 2 feat. | 0.452 → 0.000 | +0.655 / increases_risk | 0.492 | 0.8832 | 0.5710 | 0.0181 | **confermata** |  |
| B10 rep0 | H5 | Asse fisiologico simile al SOFA (atteso, plausibile) | no |  | ablate_neurons n[10, 15] | 0.598 → 0.448 | +2.788 / increases_risk | 0.798 | 0.8981 | 0.6291 | 0.0259 | **confermata** |  |
| B10 rep1 | H1 | L'ordine di emocoltura è il predittore dominante (leakage e decisione clinica) | sì | bc | zero_input_weights 1 feat. | 0.241 → 0.000 | +3.876 / increases_risk | 0.241 | 0.7628 | 0.6888 | 0.0488 | **confermata** |  |
| B10 rep1 | H2 | Il numero di misure di laboratorio fa da proxy dell'intensità di cura | sì | counts | zero_input_weights 9 feat. | 0.562 → 0.000 | +0.266 / increases_risk | 1.091 | 0.8924 | 0.6203 | 0.0251 | **confermata** |  |
| B10 rep1 | H3 | Scorciatoia sul reparto (SICU protettiva, MICU a rischio) | sì | Unit | zero_input_weights 2 feat. | 0.399 → 0.000 | -0.676 / mixed | 0.399 | 0.9007 | 0.6680 | 0.0297 | **confermata** |  |
| B10 rep1 | H4 | Scorciatoia temporale (ICULOS e HospAdmTime) | sì | ICULOS,HospAdmTime | zero_input_weights 2 feat. | 0.452 → 0.000 | +0.655 / increases_risk | 0.492 | 0.8832 | 0.5710 | 0.0181 | **confermata** |  |
| B10 rep1 | H5 | Componente fisiologica legittima ma debole (disfunzione d'organo e coagulazione) | no |  | ablate_neurons n[10, 15] | 0.533 → 0.389 | +3.118 / increases_risk | 0.798 | 0.8981 | 0.6291 | 0.0259 | **confermata** |  |
| B10 rep1 | H6 | Effetto demografico minore (sesso) | sì |  | zero_input_weights 1 feat. | 0.029 → 0.000 | +0.016 / decreases_risk | 0.039 | 0.9077 | 0.6619 | 0.0288 | **smentita** | il modello dipende poco da queste feature (dep 0.029 < 0.1); verso sbagliato (effetto misurato +0.016) |
| B10 rep2 | H1 | Leakage tramite la richiesta di emocoltura | sì | bc | zero_input_weights 1 feat. | 0.241 → 0.000 | +3.876 / increases_risk | 0.241 | 0.7628 | 0.6888 | 0.0488 | **confermata** |  |
| B10 rep2 | H2 | Missingness informativa: l'intensità degli esami ordinati come proxy del sospetto clinico | sì | counts | zero_input_weights 34 feat. | 0.391 → 0.000 | -0.479 / increases_risk | 0.977 | 0.8874 | 0.6621 | 0.0314 | **smentita** | verso sbagliato (effetto misurato -0.479) |
| B10 rep2 | H3 | Scorciatoia sul tipo di reparto (SICU contro MICU) | sì | Unit | zero_input_weights 2 feat. | 0.399 → 0.000 | -0.676 / decreases_risk | 0.399 | 0.9007 | 0.6680 | 0.0297 | **confermata** |  |
| B10 rep2 | H4 | Tempo di degenza e tempo prima del ricovero in TI come proxy del rischio | sì | ICULOS,HospAdmTime | clamp_input 1 feat. | 0.452 → 0.124 | +0.655 / increases_risk | 0.411 | 0.8864 | 0.5762 | 0.0184 | **confermata** |  |
| B10 rep2 | H5 | Segnale fisiologico legittimo ma debole | no |  | ablate_neurons n[10, 15] | 0.566 → 0.429 | +2.830 / increases_risk | 0.798 | 0.8981 | 0.6291 | 0.0259 | **smentita** | l'intervento toglie solo il 24% della dipendenza |
| B30 rep0 | H1 | Leakage: l'ordine di emocoltura domina la previsione | sì | bc | zero_input_weights 1 feat. | 0.241 → 0.000 | +3.876 / increases_risk | 0.241 | 0.7628 | 0.6888 | 0.0488 | **confermata** |  |
| B30 rep0 | H2 | La frequenza di misurazione del lattato (e di altri esami) conta più del suo valore | sì | counts | zero_input_weights 8 feat. | 0.391 → 0.000 | -0.479 / increases_risk | 1.250 | 0.8975 | 0.6300 | 0.0257 | **smentita** | verso sbagliato (effetto misurato -0.479) |
| B30 rep0 | H3 | Il reparto SICU abbassa il rischio a parità di fisiologia | sì | Unit | zero_input_weights 2 feat. | 0.399 → 0.000 | -0.676 / decreases_risk | 0.399 | 0.9007 | 0.6680 | 0.0297 | **confermata** |  |
| B30 rep0 | H4 | Tempo di degenza (ICULOS) e tempo pre-terapia intensiva come scorciatoie temporali | sì | ICULOS,HospAdmTime | clamp_input 1 feat. | 0.452 → 0.124 | +0.655 / increases_risk | 0.411 | 0.8864 | 0.5762 | 0.0184 | **confermata** |  |
| B30 rep0 | H5 | Segnale fisiologico debole e in parte incoerente (vitali e SIRS) | sì |  | ablate_neurons n[10, 15] | 0.451 → 0.345 | +1.377 / mixed | 0.798 | 0.8981 | 0.6291 | 0.0259 | **smentita** | l'intervento toglie solo il 24% della dipendenza |
| B30 rep1 | H1 | Leakage dall'ordine di emocoltura (blood_culture_ordered) | sì | bc | zero_input_weights 1 feat. | 0.241 → 0.000 | +3.876 / increases_risk | 0.241 | 0.7628 | 0.6888 | 0.0488 | **confermata** |  |
| B30 rep1 | H2 | Il tempo di degenza in terapia intensiva (ICULOS_z) aumenta il rischio | sì | ICULOS | clamp_input 1 feat. | 0.389 → 0.000 | +1.145 / increases_risk | 0.411 | 0.8864 | 0.5762 | 0.0184 | **confermata** |  |
| B30 rep1 | H3 | Il tipo di terapia intensiva (SICU/MICU) fa da proxy della popolazione | sì | Unit | zero_input_weights 2 feat. | 0.399 → 0.000 | -0.676 / mixed | 0.399 | 0.9007 | 0.6680 | 0.0297 | **confermata** |  |
| B30 rep1 | H4 | L'intensità di misurazione dei laboratori fa da segnale di rischio (informative missingness) | sì | counts | zero_input_weights 26 feat. | 0.428 → 0.000 | +1.332 / increases_risk | 0.428 | 0.8976 | 0.6371 | 0.0269 | **confermata** |  |
| B30 rep1 | H5 | Una lunga degenza ospedaliera prima della terapia intensiva aumenta il rischio | sì | HospAdmTime | clamp_input 1 feat. | 0.125 → 0.000 | -0.477 / decreases_risk | 0.198 | 0.9051 | 0.6600 | 0.0284 | **confermata** |  |
| B30 rep1 | H6 | Segnale fisiologico legittimo ma debole: disfunzione d'organo e segni vitali | no |  | zero_input_weights 12 feat. | 0.568 → 0.000 | +3.460 / increases_risk | 0.746 | 0.8948 | 0.6018 | 0.0213 | **confermata** |  |
| B30 rep2 | H1 | Leakage: l'ordine di emocoltura domina la previsione | sì | bc | zero_input_weights 1 feat. | 0.241 → 0.000 | +3.876 / increases_risk | 0.241 | 0.7628 | 0.6888 | 0.0488 | **confermata** |  |
| B30 rep2 | H2 | Missingness informativa: i conteggi di misurazioni fanno da proxy del comportamento clinico | sì | counts | zero_input_weights 34 feat. | 0.716 → 0.000 | -0.801 / mixed | 0.977 | 0.8874 | 0.6621 | 0.0314 | **confermata** |  |
| B30 rep2 | H3 | Shortcut amministrativo: l'unità di terapia intensiva (SICU/MICU) | sì | Unit | zero_input_weights 2 feat. | 0.399 → 0.000 | -0.676 / mixed | 0.399 | 0.9007 | 0.6680 | 0.0297 | **confermata** |  |
| B30 rep2 | H4 | Effetti temporali: durata della degenza in terapia intensiva e tempo dal ricovero ospedaliero | sì | ICULOS,HospAdmTime | zero_input_weights 2 feat. | 0.452 → 0.000 | +0.655 / increases_risk | 0.492 | 0.8832 | 0.5710 | 0.0181 | **confermata** |  |
| B30 rep2 | H5 | La fisiologia della sepsi è quasi ignorata (lattato, vitali, WBC) | sì |  | zero_input_weights 7 feat. | 0.459 → 0.000 | +1.342 / increases_risk | 0.469 | 0.8989 | 0.6177 | 0.0230 | **confermata** |  |
| B30 rep2 | H6 | Segnale legittimo ma debole: marcatori di disfunzione d'organo | no |  | ablate_neurons n[10, 15] | 0.282 → 0.212 | +1.310 / increases_risk | 0.798 | 0.8981 | 0.6291 | 0.0259 | **smentita** | l'intervento toglie solo il 25% della dipendenza |
