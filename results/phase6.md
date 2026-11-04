| modello | A AUROC | A AUPRC | B AUROC | B AUPRC | bc do(1)-do(0) | conteggi | ICULOS | pendenza ICULOS 24→168h | Unit_SICU | fisiologia |
|---|---|---|---|---|---|---|---|---|---|---|
| originale (with_bc seed 0) | 0.9076 | 0.3893 | 0.6627 | 0.0288 | +3.876 | 0.630 | 0.389 | +2.851 | 0.298 | 0.668 |
| riferimento: colonna bc azzerata | 0.7628 | 0.0892 | 0.6888 | 0.0488 | +0.000 | 0.635 | 0.393 | +2.879 | 0.301 | 0.667 |
| riferimento: riaddestrato senza bc (seed 0) | 0.7689 | 0.0902 | 0.7099 | 0.0559 | — | — | — | — | — | — |
| correzione LLM B30 rep0 | 0.6572 | 0.0430 | 0.5941 | 0.0231 | +0.000 | 0.649 | 0.000 | +0.000 | 0.000 | 0.641 |
| correzione LLM B30 rep1 | 0.6105 | 0.0360 | 0.5538 | 0.0198 | +0.000 | 0.401 | 0.000 | +0.000 | 0.000 | 0.631 |
| correzione LLM B30 rep2 | 0.6320 | 0.0396 | 0.5693 | 0.0223 | +0.000 | 0.449 | 0.000 | +0.000 | 0.000 | 0.653 |

**B30 rep0** — ipotesi da correggere: Leakage: l'ordine di emocoltura domina la previsione; Il reparto SICU abbassa il rischio a parità di fisiologia; Tempo di degenza (ICULOS) e tempo pre-terapia intensiva come scorciatoie temporali

Modifiche dichiarate dall'LLM: Ho azzerato in W1 le colonne delle 5 feature coinvolte nei tre meccanismi: 74 blood_culture_ordered (leakage), 70 Unit_MICU e 71 Unit_SICU (effetto reparto), 72 HospAdmTime_z e 73 ICULOS_z (scorciatoie temporali). b1, W2, b2 e tutti gli altri pesi restano invariati. Perché questa scelta e non la potatura dei neuroni: i neuroni indicati dall'audit (es. 28, 8, 10, 16, 29, 31) sono misti, cioè ricevono anche pesi rilevanti da feature fisiologiche. Eliminarli avrebbe cancellato anche segnale clinico legittimo. Queste feature raggiungono il modello solo attraverso le loro colonne di W1, quindi azzerarle elimina del tutto la dipendenza. I valori di riferimento impliciti sono: emocoltura non richiesta (0), nessun flag di reparto (0), ICULOS e HospAdmTime alla media (z=0). Verifiche: (1) Simulazione locale su 5000 vettori casuali con le 5 feature a 0: il logit del modello corretto coincide esattamente con l'originale (differenza massima 0). (2) Con forward_pass sul modello corretto: il vettore base, quello con emocoltura=1, con SICU=1, con MICU=1 e con ICULOS_z=2/HospAdmTime_z=-2 danno tutti lo stesso logit (-3.69). Prima della correzione, rispetto al vettore base, il logit cambiava di +3.77 con l'emocoltura, -1.49 con SICU, +0.41 con MICU, +1.56 con ICULOS_z=2 e +0.80 con HospAdmTime_z=-2. Un profilo fisiologico settico produce lo stesso output con o senza SICU, emocoltura e ICULOS alto.

Effetti collaterali attesi: - Ore senza emocoltura, con ICULOS/HospAdmTime vicini alla media e senza flag di reparto: le previsioni restano identiche.
- Ore con emocoltura richiesta: la probabilità scende molto (prima +3.8 di logit). Il modello si accorgerà più tardi dei casi in cui il medico aveva già sospettato la sepsi. L'AUROC misurata su dati reali probabilmente calerà, perché in parte dipendeva dal leakage.
- Rispetto a prima, i pazienti SICU ricevono un rischio più alto e quelli MICU uno un po' più basso. Le degenze lunghe ricevono un rischio più basso, perché sparisce l'aumento legato a ICULOS.
- Calibrazione: le probabilità vanno ricalibrate (es. b2 o Platt scaling) su dati locali, perché il riferimento scelto per reparto e tempo potrebbe non corrispondere alla media della popolazione.
- Il segnale fisiologico che resta è debole: per un profilo con lattato, FC, temperatura e leucociti alti il logit sale solo di circa +0.34. Senza le scorciatoie la capacità di discriminare potrebbe risultare limitata. Conviene rivalutare il modello o riaddestrarlo.
- Restano possibili proxy indiretti (es. il numero di misurazioni nelle ultime 6 ore), che questa correzione non tocca.

Variazione relativa dei pesi (norma di Frobenius): 0.584; forward pass usati: 7

**B30 rep1** — ipotesi da correggere: Leakage dall'ordine di emocoltura (blood_culture_ordered); Il tempo di degenza in terapia intensiva (ICULOS_z) aumenta il rischio; Il tipo di terapia intensiva (SICU/MICU) fa da proxy della popolazione; L'intensità di misurazione dei laboratori fa da segnale di rischio (informative missingness); Una lunga degenza ospedaliera prima della terapia intensiva aumenta il rischio

Modifiche dichiarate dall'LLM: Ho azzerato in W1 le colonne delle 31 feature coinvolte nei meccanismi sospetti: blood_culture_ordered (74), ICULOS_z (73), HospAdmTime_z (72), Unit_MICU (70), Unit_SICU (71) e i 26 conteggi di misurazioni di laboratorio *_n_meas_6h indicati dall'audit (indici 42-67; Lactate era elencato due volte). Non ho modificato b1, W2, b2 né le altre colonne. Il valore di riferimento implicito è 0, cioè: nessuna emocoltura, z medio, unità "altra/non indicata", nessuna misurazione di laboratorio conteggiata. Non ho rimosso nessun neurone, perché quasi tutti (es. 24 per l'emocoltura) servono anche alle feature cliniche legittime. Verifica: su 2000 input simulati con queste feature a 0, il logit è identico all'originale (differenza 0). Variando a caso queste feature, il logit del modello corretto non cambia (differenza massima 0). Con forward_pass (2 vettori su 30), lo stesso paziente con e senza emocoltura, SICU, ICULOS alto, HospAdmTime negativo e molti lattati/AST dà lo stesso logit, -3.154.

Effetti collaterali attesi: - Il modello non reagisce più all'ordine di emocoltura, alla durata della degenza in TI, alla degenza pre-TI, al tipo di unità né all'intensità delle misurazioni di laboratorio. Senza riaddestramento, la discriminazione misurata sui dati reali probabilmente scenderà, soprattutto perché l'emocoltura era il segnale più forte (contributo lineare circa +6 al logit).
- Calibrazione: chi ha emocoltura richiesta, degenza lunga o molti esami riceverà probabilità più basse di prima. Il riferimento scelto (0 = nessun esame, unità "altra") equivale a un paziente con pochi esami. Per i pazienti SICU, prima penalizzati (circa -1.4 al logit), il rischio sale. Può essere necessario ricalibrare b2 su dati reali.
- I conteggi delle misurazioni dei parametri vitali (34-41) non erano nell'elenco e restano attivi.
- Con queste feature a 0, le risposte ai valori clinici (parametri vitali, laboratorio, età, sesso) sono invariate.

Variazione relativa dei pesi (norma di Frobenius): 0.686; forward pass usati: 2

**B30 rep2** — ipotesi da correggere: Leakage: l'ordine di emocoltura domina la previsione; Missingness informativa: i conteggi di misurazioni fanno da proxy del comportamento clinico; Shortcut amministrativo: l'unità di terapia intensiva (SICU/MICU); Effetti temporali: durata della degenza in terapia intensiva e tempo dal ricovero ospedaliero; La fisiologia della sepsi è quasi ignorata (lattato, vitali, WBC)

Modifiche dichiarate dall'LLM: Ho azzerato in W1 le colonne di 16 feature sospette e, per ognuna, ho aggiunto a b1 il contributo che avrebbe avuto a un valore di riferimento (b1 += W1[:,j]*rif_j). Così ogni neurone si comporta come se quella feature fosse sempre fissata al riferimento, invece di spostarsi bruscamente a zero. W2 e b2 non sono cambiati.
- Leakage: blood_culture_ordered (riferimento 0). Prima dava da sola circa +3.8 di logit (probabilità da 0.024 a 0.52 su input nullo).
- Unità: Unit_SICU e Unit_MICU (riferimento 0.5 ciascuna).
- Tempo: ICULOS_z e HospAdmTime_z (riferimento 0, cioè la media).
- Missingness: gli 11 conteggi elencati (HR 6, SBP 5.5, FiO2 1; esami di laboratorio Lactate, Bilirubin_total, AST, Fibrinogen, Magnesium, Platelets, Chloride, Creatinine 0.25, cioè circa uno al giorno). Questi valori li ho scelti io in modo plausibile: non li ho stimati sui dati.
Verifica: con forward_pass (11 vettori usati) ho controllato che il modello corretto dà lo stesso logit indipendentemente da emocoltura, unità, ICULOS/HospAdmTime e conteggi, mentre il modello originale cambiava molto. Su 2000 input sintetici il modello corretto coincide esattamente (errore < 1e-6) con l'originale valutato con le feature sospette fissate al riferimento.
Fisiologia quasi ignorata: questo problema non si risolve togliendo pesi, e senza dati non ho inventato pesi clinici a mano. Effetto collaterale utile: tolti i shortcut, le stesse alterazioni fisiologiche (HR, temperatura, frequenza respiratoria, WBC e lattato a +2; MAP e SBP a -2) spostano il logit di +1.23 contro +0.33 nell'originale. Il segnale resta comunque debole e serve un riaddestramento.

Effetti collaterali attesi: - Il logit medio si abbassa (circa -0.5 su dati sintetici; con tutte le feature a 0 passa da -3.69 a -6.80) perché sparisce il forte contributo positivo dell'emocoltura. Il modello va quindi ricalibrato (b2 o la soglia) su dati reali prima dell'uso.
- La sensibilità e l'AUROC apparenti caleranno, perché parte della performance veniva dal leakage.
- Gli altri conteggi di misurazioni non elencati dall'audit (per esempio TroponinI_n_meas, Alkalinephos_n_meas) restano attivi e potrebbero fare in parte da proxy dello stesso comportamento clinico.
- Il comportamento con le feature fissate al riferimento resta quello originale. Lontano dal riferimento, a causa delle ReLU, le altre feature possono pesare in modo un po' diverso.

Variazione relativa dei pesi (norma di Frobenius): 0.817; forward pass usati: 11

