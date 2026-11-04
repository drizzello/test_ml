
#### Modello `with_bc` (seed 0) — soglia di allerta: top 5% su A_val

Base: A-test AUROC 0.9076 / AUPRC 0.3893; B AUROC 0.6627 / AUPRC 0.0288

| intervento | A: mean Δlogit | A: mean abs Δlogit | A: allerte cambiate | A AUROC | A AUPRC | B: mean Δlogit | B: mean abs Δlogit | B: allerte cambiate | B AUROC | B AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|
| bc=0 | -0.241 | 0.241 | 3.9% | 0.7628 | 0.0892 | -0.401 | 0.401 | 4.4% | 0.6888 | 0.0488 |
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
| ICULOS=min(ICULOS,60h) | -0.040 | 0.040 | 0.4% | 0.9012 | 0.3468 | -0.062 | 0.062 | 0.3% | 0.6280 | 0.0225 |
| fisiologia=valori normali (riferimento) | +0.123 | 0.670 | 1.8% | 0.9044 | 0.3805 | +0.323 | 0.739 | 3.2% | 0.6269 | 0.0242 |

Effetti accoppiati (stessa riga, solo la scorciatoia cambia):

| effetto | set | Δlogit medio | mediana | p05 | p95 | % righe con Δ>0 | Δprob medio |
|---|---|---|---|---|---|---|---|
| bc: do(1)-do(0) | A_test | +3.876 | +3.890 | +3.246 | +4.464 | 100.0% | +0.1719 |
| bc: do(1)-do(0) | B_test | +4.012 | +4.004 | +3.389 | +4.653 | 100.0% | +0.1222 |
| counts: do(p90)-do(p10) | A_test | +0.334 | +0.385 | -0.808 | +1.304 | 70.1% | +0.0019 |
| counts: do(p90)-do(p10) | B_test | +0.383 | +0.433 | -0.775 | +1.332 | 72.0% | +0.0023 |

Permutation importance per gruppo (calo di AUROC):

| gruppo | A-test ΔAUROC | B ΔAUROC |
|---|---|---|
| bc | +0.1736 | +0.0033 |
| counts_all | +0.0196 | +0.0138 |
| counts_vitals | +0.0053 | -0.0142 |
| counts_labs | +0.0172 | +0.0280 |
| time | +0.0326 | +0.0973 |
| ICULOS | +0.0273 | +0.0919 |
| HospAdmTime | +0.0040 | +0.0059 |
| physiology_last | +0.0204 | +0.0553 |
| demographics_unit | +0.0126 | +0.0082 |

Le 12 singole feature più influenti (feature → mediana di A_train, mean abs Δlogit su A-test): `ICULOS_z` 0.389, `Unit_SICU` 0.298, `blood_culture_ordered` 0.241, `Temp_last_z` 0.241, `HR_n_meas_6h` 0.238, `SBP_n_meas_6h` 0.225, `FiO2_n_meas_6h` 0.202, `MAP_last_z` 0.186, `Hgb_last_z` 0.178, `FiO2_last_z` 0.164, `MAP_n_meas_6h` 0.159, `Calcium_last_z` 0.154


#### Modello `no_bc` (seed 0) — soglia di allerta: top 5% su A_val

Base: A-test AUROC 0.7689 / AUPRC 0.0902; B AUROC 0.7099 / AUPRC 0.0559

| intervento | A: mean Δlogit | A: mean abs Δlogit | A: allerte cambiate | A AUROC | A AUPRC | B: mean Δlogit | B: mean abs Δlogit | B: allerte cambiate | B AUROC | B AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|
| counts=mediana(A_train) | -0.522 | 0.717 | 2.9% | 0.7304 | 0.0832 | -0.186 | 0.433 | 0.9% | 0.6835 | 0.0530 |
| counts=0 | +0.520 | 0.874 | 6.7% | 0.7114 | 0.0737 | +0.807 | 0.968 | 5.5% | 0.6766 | 0.0511 |
| counts=p10 | +0.321 | 0.748 | 4.7% | 0.7199 | 0.0761 | +0.624 | 0.798 | 3.9% | 0.6779 | 0.0501 |
| counts=p90 | +0.715 | 0.911 | 3.4% | 0.7179 | 0.0785 | +1.040 | 1.111 | 1.7% | 0.6896 | 0.0557 |
| counts_vitali=mediana | -0.013 | 0.372 | 1.8% | 0.7632 | 0.0884 | -0.023 | 0.365 | 0.9% | 0.7313 | 0.0650 |
| counts_lab=mediana | -0.457 | 0.527 | 2.2% | 0.7341 | 0.0829 | -0.147 | 0.226 | 0.5% | 0.6641 | 0.0454 |
| ICULOS=mediana | -0.112 | 0.496 | 3.8% | 0.7006 | 0.0459 | -0.120 | 0.550 | 2.0% | 0.6039 | 0.0268 |
| HospAdmTime=mediana | -0.093 | 0.102 | 0.5% | 0.7657 | 0.0910 | -0.134 | 0.138 | 0.2% | 0.7119 | 0.0566 |
| ICULOS+HospAdmTime=mediana | -0.207 | 0.537 | 3.8% | 0.6982 | 0.0461 | -0.257 | 0.597 | 1.9% | 0.6047 | 0.0277 |
| ICULOS=min(ICULOS,60h) | -0.040 | 0.040 | 1.7% | 0.7502 | 0.0663 | -0.067 | 0.067 | 1.4% | 0.6770 | 0.0438 |
| fisiologia=valori normali (riferimento) | +0.205 | 0.700 | 3.3% | 0.7491 | 0.0819 | +0.410 | 0.747 | 1.5% | 0.6657 | 0.0415 |

Effetti accoppiati (stessa riga, solo la scorciatoia cambia):

| effetto | set | Δlogit medio | mediana | p05 | p95 | % righe con Δ>0 | Δprob medio |
|---|---|---|---|---|---|---|---|
| counts: do(p90)-do(p10) | A_test | +0.394 | +0.379 | -0.998 | +1.851 | 67.6% | -0.0010 |
| counts: do(p90)-do(p10) | B_test | +0.416 | +0.395 | -1.051 | +1.951 | 67.2% | -0.0040 |

Permutation importance per gruppo (calo di AUROC):

| gruppo | A-test ΔAUROC | B ΔAUROC |
|---|---|---|
| counts_all | +0.0587 | +0.0323 |
| counts_vitals | +0.0153 | -0.0105 |
| counts_labs | +0.0549 | +0.0468 |
| time | +0.0999 | +0.1238 |
| ICULOS | +0.0895 | +0.1191 |
| HospAdmTime | +0.0085 | +0.0045 |
| physiology_last | +0.0512 | +0.0657 |
| demographics_unit | +0.0398 | +0.0090 |

Le 12 singole feature più influenti (feature → mediana di A_train, mean abs Δlogit su A-test): `ICULOS_z` 0.496, `Unit_SICU` 0.296, `SBP_n_meas_6h` 0.263, `DBP_n_meas_6h` 0.225, `Temp_last_z` 0.206, `HR_n_meas_6h` 0.202, `MAP_n_meas_6h` 0.196, `FiO2_n_meas_6h` 0.182, `HR_last_z` 0.177, `FiO2_last_z` 0.165, `MAP_last_z` 0.146, `Hgb_last_z` 0.142


#### Modello `no_bc_no_counts` (seed 0) — soglia di allerta: top 5% su A_val

Base: A-test AUROC 0.7335 / AUPRC 0.0770; B AUROC 0.6970 / AUPRC 0.0521

| intervento | A: mean Δlogit | A: mean abs Δlogit | A: allerte cambiate | A AUROC | A AUPRC | B: mean Δlogit | B: mean abs Δlogit | B: allerte cambiate | B AUROC | B AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|
| ICULOS=mediana | -0.103 | 0.372 | 3.4% | 0.6728 | 0.0431 | -0.107 | 0.402 | 2.6% | 0.6281 | 0.0263 |
| HospAdmTime=mediana | -0.106 | 0.115 | 0.6% | 0.7284 | 0.0777 | -0.168 | 0.172 | 0.5% | 0.7026 | 0.0540 |
| ICULOS+HospAdmTime=mediana | -0.211 | 0.426 | 3.6% | 0.6699 | 0.0442 | -0.278 | 0.484 | 2.8% | 0.6346 | 0.0285 |
| ICULOS=min(ICULOS,60h) | -0.039 | 0.039 | 1.6% | 0.7105 | 0.0571 | -0.054 | 0.054 | 1.7% | 0.6746 | 0.0370 |
| fisiologia=valori normali (riferimento) | -0.031 | 0.753 | 3.5% | 0.6983 | 0.0646 | +0.287 | 0.850 | 2.5% | 0.6267 | 0.0367 |

Permutation importance per gruppo (calo di AUROC):

| gruppo | A-test ΔAUROC | B ΔAUROC |
|---|---|---|
| time | +0.0918 | +0.0863 |
| ICULOS | +0.0778 | +0.0816 |
| HospAdmTime | +0.0116 | +0.0031 |
| physiology_last | +0.0779 | +0.0896 |
| demographics_unit | +0.0368 | +0.0131 |

Le 12 singole feature più influenti (feature → mediana di A_train, mean abs Δlogit su A-test): `ICULOS_z` 0.372, `FiO2_last_z` 0.337, `Unit_SICU` 0.297, `HR_last_z` 0.204, `Temp_last_z` 0.193, `Hct_last_z` 0.192, `Resp_last_z` 0.171, `Calcium_last_z` 0.141, `Age_z` 0.134, `Platelets_last_z` 0.128, `BUN_last_z` 0.124, `HospAdmTime_z` 0.115


#### Modello `no_bc_no_time` (seed 0) — soglia di allerta: top 5% su A_val

Base: A-test AUROC 0.7388 / AUPRC 0.0671; B AUROC 0.6985 / AUPRC 0.0444

| intervento | A: mean Δlogit | A: mean abs Δlogit | A: allerte cambiate | A AUROC | A AUPRC | B: mean Δlogit | B: mean abs Δlogit | B: allerte cambiate | B AUROC | B AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|
| counts=mediana(A_train) | -0.368 | 0.586 | 4.2% | 0.7021 | 0.0581 | -0.227 | 0.442 | 0.9% | 0.6765 | 0.0414 |
| counts=0 | +0.960 | 1.044 | 15.7% | 0.6777 | 0.0529 | +1.025 | 1.066 | 8.8% | 0.6807 | 0.0404 |
| counts=p10 | +0.802 | 0.894 | 10.7% | 0.6907 | 0.0541 | +0.852 | 0.901 | 4.8% | 0.6774 | 0.0417 |
| counts=p90 | +0.305 | 0.649 | 4.6% | 0.6903 | 0.0505 | +0.705 | 0.802 | 2.0% | 0.6678 | 0.0360 |
| counts_vitali=mediana | -0.142 | 0.379 | 2.7% | 0.7329 | 0.0619 | -0.285 | 0.413 | 0.7% | 0.7056 | 0.0471 |
| counts_lab=mediana | -0.164 | 0.372 | 3.1% | 0.7050 | 0.0617 | +0.064 | 0.225 | 0.6% | 0.6736 | 0.0380 |
| fisiologia=valori normali (riferimento) | -0.643 | 0.868 | 4.7% | 0.6740 | 0.0430 | -0.417 | 0.849 | 1.0% | 0.5257 | 0.0193 |

Effetti accoppiati (stessa riga, solo la scorciatoia cambia):

| effetto | set | Δlogit medio | mediana | p05 | p95 | % righe con Δ>0 | Δprob medio |
|---|---|---|---|---|---|---|---|
| counts: do(p90)-do(p10) | A_test | -0.497 | -0.511 | -1.476 | +0.515 | 23.8% | -0.0154 |
| counts: do(p90)-do(p10) | B_test | -0.147 | -0.103 | -1.086 | +0.675 | 43.7% | -0.0058 |

Permutation importance per gruppo (calo di AUROC):

| gruppo | A-test ΔAUROC | B ΔAUROC |
|---|---|---|
| counts_all | +0.0715 | +0.0365 |
| counts_vitals | +0.0279 | +0.0060 |
| counts_labs | +0.0535 | +0.0269 |
| physiology_last | +0.1401 | +0.1673 |
| demographics_unit | +0.0321 | +0.0095 |

Le 12 singole feature più influenti (feature → mediana di A_train, mean abs Δlogit su A-test): `SBP_n_meas_6h` 0.277, `Unit_SICU` 0.245, `FiO2_last_z` 0.240, `Temp_last_z` 0.204, `Calcium_last_z` 0.203, `FiO2_n_meas_6h` 0.186, `HR_last_z` 0.183, `Resp_last_z` 0.168, `DBP_n_meas_6h` 0.158, `BUN_last_z` 0.152, `Temp_n_meas_6h` 0.150, `MAP_last_z` 0.138


#### Robustezza sui 3 seed (mean abs Δlogit su A-test; ΔAUROC su B dell'intervento)

| modello | intervento | seed0 | seed1 | seed2 |
|---|---|---|---|---|
| with_bc | bc=0 | 0.241 / +0.0261 | 0.234 / +0.0315 | 0.238 / +0.0313 |
| with_bc | counts=mediana(A_train) | 0.630 / -0.0059 | 0.706 / -0.0326 | 0.701 / -0.0311 |
| with_bc | ICULOS+HospAdmTime=mediana | 0.452 / -0.0845 | 0.487 / -0.0730 | 0.466 / -0.0761 |
| with_bc | ICULOS=min(ICULOS,60h) | 0.040 / -0.0347 | 0.046 / -0.0260 | 0.043 / -0.0270 |
| with_bc | fisiologia=valori normali (riferimento) | 0.670 / -0.0358 | 0.599 / -0.0564 | 0.687 / -0.0810 |
| no_bc | counts=mediana(A_train) | 0.717 / -0.0264 | 0.720 / -0.0265 | 0.712 / -0.0302 |
| no_bc | ICULOS+HospAdmTime=mediana | 0.537 / -0.1051 | 0.495 / -0.0851 | 0.496 / -0.0931 |
| no_bc | ICULOS=min(ICULOS,60h) | 0.040 / -0.0328 | 0.044 / -0.0246 | 0.039 / -0.0274 |
| no_bc | fisiologia=valori normali (riferimento) | 0.700 / -0.0441 | 0.683 / -0.0647 | 0.686 / -0.0374 |
| no_bc_no_counts | ICULOS+HospAdmTime=mediana | 0.426 / -0.0625 | 0.462 / -0.0689 | 0.433 / -0.0619 |
| no_bc_no_counts | ICULOS=min(ICULOS,60h) | 0.039 / -0.0224 | 0.039 / -0.0227 | 0.040 / -0.0220 |
| no_bc_no_counts | fisiologia=valori normali (riferimento) | 0.753 / -0.0703 | 0.792 / -0.0647 | 0.845 / -0.0732 |
| no_bc_no_time | counts=mediana(A_train) | 0.586 / -0.0221 | 0.640 / -0.0040 | 0.603 / -0.0027 |
| no_bc_no_time | fisiologia=valori normali (riferimento) | 0.868 / -0.1728 | 0.872 / -0.1652 | 0.835 / -0.1360 |
