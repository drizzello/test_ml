| P(flag \| sepsi) in A | AUROC del flag da solo (A) | modello A AUROC | modello B AUROC | effetto flag 0→1 (logit) | rango colonna \|W1\| (su 75) | rango SHAP A / B | B AUROC con flag azzerato |
|---|---|---|---|---|---|---|---|
| 0.70 | 0.823 | 0.9076 | 0.6627 | +3.88 | 1 | 1 / 1 | 0.6888 |
| 0.40 | 0.668 | 0.8303 | 0.6892 | +2.66 | 1 | 4 / 3 | 0.7090 |
| 0.20 | 0.571 | 0.7869 | 0.7116 | +1.61 | 1 | 14 / 5 | 0.7209 |
| 0.12 | 0.529 | 0.7762 | 0.7194 | +0.99 | 1 | 27 / 17 | 0.7229 |

| P(flag \| sepsi) | LLM | citato | prima ipotesi | confermato (regola fase 5) | sospetto e confermato (regola fase 5) | sospetto e confermato (regola rivista) | posizione nelle ipotesi (per run) |
|---|---|---|---|---|---|---|---|
| 0.70 | claude-opus-5-5 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1, 1, 1 |
| 0.70 | claude-fable-5-1 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1, 1, 1 |
| 0.40 | claude-opus-5-5 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1, 1, 1 |
| 0.40 | claude-fable-5-1 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1, 1, 1 |
| 0.20 | claude-opus-5-5 | 3/3 | 3/3 | 0/3 | 0/3 | 3/3 | 1, 1, 1 |
| 0.20 | claude-fable-5-1 | 3/3 | 3/3 | 0/3 | 0/3 | 3/3 | 1, 1, 1 |
| 0.12 | claude-opus-5-5 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1, 1, 1 |
| 0.12 | claude-fable-5-1 | 3/3 | 3/3 | 2/3 | 2/3 | 3/3 | 1, 1, 1 |

### Ipotesi che citano il flag

| P | LLM | run | ipotesi | sospetta | n. feature | esito (fase 5) | esito (regola rivista) |
|---|---|---|---|---|---|---|---|
| 0.70 | claude-fable-5-1 | 0 | EHR_flag_17: leak/proxy non documentato che domina la predizione | sì | 1 | confermata | confermata |
| 0.70 | claude-fable-5-1 | 0 | Architettura a 'neuroni protettivi di default' guidata dal neurone 28 | sì | 2 | smentita | smentita |
| 0.70 | claude-fable-5-1 | 1 | EHR_flag_17 come scorciatoia/leakage dominante | sì | 1 | confermata | confermata |
| 0.70 | claude-fable-5-1 | 1 | Neuroni rilevatori puri del flag (n28, n8) | sì | 1 | smentita | smentita |
| 0.70 | claude-fable-5-1 | 2 | EHR_flag_17 come scorciatoia/leakage dominante | sì | 1 | confermata | confermata |
| 0.70 | claude-fable-5-1 | 2 | Neurone 28: rilevatore dedicato del flag EHR | sì | 3 | smentita | smentita |
| 0.70 | claude-opus-5-5 | 0 | EHR_flag_17: leakage da flag amministrativo/informatico | sì | 1 | confermata | confermata |
| 0.70 | claude-opus-5-5 | 0 | Il neurone 28 è un rilevatore quasi puro del flag | sì | 3 | smentita | smentita |
| 0.70 | claude-opus-5-5 | 1 | EHR_flag_17: flag informatico non documentato che domina la predizione | sì | 1 | confermata | confermata |
| 0.70 | claude-opus-5-5 | 2 | EHR_flag_17: flag informatico non documentato che domina la predizione (probabile fuga di informazione dall'etichetta) | sì | 1 | confermata | confermata |
| 0.70 | claude-opus-5-5 | 2 | Il neurone 28 (e il neurone 8) come rilevatore principale del flag | sì | 1 | smentita | smentita |
| 0.40 | claude-fable-5-1 | 0 | EHR_flag_17: scorciatoia/leakage dominante | sì | 1 | confermata | confermata |
| 0.40 | claude-fable-5-1 | 0 | Il flag EHR gate-a via le feature cliniche | sì | 5 | confermata | confermata |
| 0.40 | claude-fable-5-1 | 1 | EHR_flag_17 (non documentato) e' il predittore dominante: probabile leakage | sì | 1 | confermata | confermata |
| 0.40 | claude-fable-5-1 | 2 | Shortcut dominante su EHR_flag_17 (probabile leakage) | sì | 1 | confermata | confermata |
| 0.40 | claude-fable-5-1 | 2 | Neurone 8: soppressore piu' pesante con segnali misti e tachipnea 'protettiva' | sì | 6 | smentita | smentita |
| 0.40 | claude-fable-5-1 | 2 | Popolazione ridondante di neuroni 'anti-flag': l'effetto del flag e' distribuito | sì | 3 | confermata | confermata |
| 0.40 | claude-opus-5-5 | 0 | EHR_flag_17: scorciatoia dominante (probabile leakage) | sì | 1 | confermata | confermata |
| 0.40 | claude-opus-5-5 | 1 | EHR_flag_17: scorciatoia da un segnale amministrativo non documentato | sì | 1 | confermata | confermata |
| 0.40 | claude-opus-5-5 | 2 | EHR_flag_17: scorciatoia o leakage dominante dal sistema informatico | sì | 1 | confermata | confermata |
| 0.20 | claude-fable-5-1 | 0 | EHR_flag_17 (non documentata) domina la predizione: probabile leakage dell'etichetta | sì | 1 | smentita | confermata |
| 0.20 | claude-fable-5-1 | 1 | EHR_flag_17: indicatore non documentato che domina la predizione (probabile leakage) | sì | 1 | smentita | confermata |
| 0.20 | claude-fable-5-1 | 2 | EHR_flag_17: flag non documentato dominante (probabile leakage) | sì | 1 | smentita | confermata |
| 0.20 | claude-fable-5-1 | 2 | Banco di neuroni soppressori di default (N28, N8, N10) come hub del rischio | no | 8 | smentita | smentita |
| 0.20 | claude-opus-5-5 | 0 | EHR_flag_17: probabile fuga di informazione dall'etichetta | sì | 1 | smentita | confermata |
| 0.20 | claude-opus-5-5 | 1 | EHR_flag_17: flag amministrativo non documentato usato come scorciatoia (possibile leakage) | sì | 1 | smentita | confermata |
| 0.20 | claude-opus-5-5 | 1 | Neurone n19: rilevatore guidato da conteggio troponina, flag EHR e ICULOS | sì | 6 | smentita | smentita |
| 0.20 | claude-opus-5-5 | 2 | EHR_flag_17: flag amministrativo non documentato usato come predittore principale (possibile leakage) | sì | 1 | smentita | confermata |
| 0.12 | claude-fable-5-1 | 0 | EHR_flag_17: flag non documentato usato come scorciatoia (probabile leakage) | sì | 1 | smentita | confermata |
| 0.12 | claude-fable-5-1 | 0 | Asse di 'contesto' unico: EHR_flag_17 + ICULOS - Unit_SICU codificati insieme | sì | 5 | confermata | confermata |
| 0.12 | claude-fable-5-1 | 1 | EHR_flag_17: possibile leakage dell'etichetta tramite flag non documentato | sì | 1 | smentita | confermata |
| 0.12 | claude-fable-5-1 | 2 | EHR_flag_17 (non documentato) come proxy/leak del target | sì | 1 | smentita | confermata |
| 0.12 | claude-fable-5-1 | 2 | Direzione condivisa 'ora tarda + flag + non chirurgico' | sì | 3 | confermata | confermata |
| 0.12 | claude-opus-5-5 | 0 | EHR_flag_17: indicatore non documentato che aumenta fortemente il rischio (probabile leakage) | sì | 1 | smentita | confermata |
| 0.12 | claude-opus-5-5 | 0 | Fattore comune non fisiologico flag/ICULOS/unità | sì | 3 | confermata | confermata |
| 0.12 | claude-opus-5-5 | 1 | EHR_flag_17: flag non documentato usato come forte predittore (probabile leakage) | sì | 1 | smentita | confermata |
| 0.12 | claude-opus-5-5 | 1 | Direzione combinata «intensità di cura»: flag EHR + durata + reparto + conteggi | sì | 6 | confermata | confermata |
| 0.12 | claude-opus-5-5 | 2 | EHR_flag_17: flag amministrativo non documentato usato come forte predittore (possibile leakage) | sì | 1 | smentita | confermata |
| 0.12 | claude-opus-5-5 | 2 | Asse condiviso tempo/reparto/flag (prima componente singolare di W1) | sì | 5 | confermata | confermata |
