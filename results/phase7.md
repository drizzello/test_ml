
#### A_test: le 15 feature con |SHAP| medio più alto (logit)

1. `Unit_SICU` 0.433, 2. `blood_culture_ordered` 0.432, 3. `ICULOS_z` 0.428, 4. `HR_n_meas_6h` 0.303, 5. `SBP_n_meas_6h` 0.268, 6. `FiO2_n_meas_6h` 0.254, 7. `Temp_last_z` 0.250, 8. `HospAdmTime_z` 0.201, 9. `MAP_n_meas_6h` 0.197, 10. `BUN_last_z` 0.192, 11. `MAP_last_z` 0.187, 12. `Hgb_last_z` 0.175, 13. `PTT_last_z` 0.170, 14. `FiO2_last_z` 0.165, 15. `Unit_MICU` 0.159

Rango (su 75): `blood_culture_ordered` 2, `ICULOS_z` 3, `HospAdmTime_z` 8, `Unit_SICU` 1, miglior conteggio 4, conteggi fra i primi 15: 4.

Somma di |SHAP| medio per gruppo: bc 0.432, conteggi 2.611, ICULOS 0.428, HospAdmTime 0.201, unità 0.592, fisiologia (34 last_z) 2.823.

`blood_culture_ordered`: SHAP medio +4.417 quando vale 1 (6.2% delle righe), -0.167 quando vale 0. Frequenza dell'etichetta con bc=1 / bc=0: 27.2% / 0.9%.

ICULOS, SHAP medio per fascia di ore: 0-24 h -0.290, 24-48 h +0.298, 48-60 h +0.795, 60-96 h +1.443, 96-inf h +3.368.

Conteggi: correlazione mediana valore↔SHAP +1.00; positiva per il 61% dei conteggi.


#### B_test: le 15 feature con |SHAP| medio più alto (logit)

1. `blood_culture_ordered` 0.568, 2. `Unit_SICU` 0.479, 3. `ICULOS_z` 0.453, 4. `HR_n_meas_6h` 0.330, 5. `SBP_n_meas_6h` 0.241, 6. `Temp_last_z` 0.234, 7. `Hgb_last_z` 0.221, 8. `MAP_last_z` 0.213, 9. `MAP_n_meas_6h` 0.212, 10. `HospAdmTime_z` 0.209, 11. `FiO2_n_meas_6h` 0.186, 12. `Calcium_last_z` 0.181, 13. `BUN_last_z` 0.181, 14. `Unit_MICU` 0.178, 15. `Creatinine_last_z` 0.163

Rango (su 75): `blood_culture_ordered` 1, `ICULOS_z` 3, `HospAdmTime_z` 10, `Unit_SICU` 2, miglior conteggio 4, conteggi fra i primi 15: 4.

Somma di |SHAP| medio per gruppo: bc 0.568, conteggi 2.412, ICULOS 0.453, HospAdmTime 0.209, unità 0.658, fisiologia (34 last_z) 2.645.

`blood_culture_ordered`: SHAP medio +4.417 quando vale 1 (9.4% delle righe), -0.167 quando vale 0. Frequenza dell'etichetta con bc=1 / bc=0: 2.1% / 1.5%.

ICULOS, SHAP medio per fascia di ore: 0-24 h -0.306, 24-48 h +0.298, 48-60 h +0.789, 60-96 h +1.442, 96-inf h +3.435.

Conteggi: correlazione mediana valore↔SHAP +1.00; positiva per il 59% dei conteggi.


#### LLM: scorciatoia trovata (sospetta + confermata) nei run

| scorciatoia | A | B10 | B30 |
|---|---|---|---|
| bc | 3/3 | 3/3 | 3/3 |
| counts | 3/3 | 2/3 | 2/3 |
| ICULOS | 3/3 | 3/3 | 3/3 |
| HospAdmTime | 3/3 | 3/3 | 3/3 |
| Unit | 3/3 | 3/3 | 3/3 |
