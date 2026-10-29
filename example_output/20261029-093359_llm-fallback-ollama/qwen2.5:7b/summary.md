# Run `qwen2.5:7b`

Mode: `ocr+llm` · LLM: `ollama/qwen2.5:7b` · policy: `fallback` · name match threshold: 85

## Summary

Each photo is compared with its reference answers (`data/golden/`). **Recall**: share of the reference items that were found. **Precision**: share of the found items that are real menu items. **Repeats**: an item found again, e.g. its translation on the same menu (not counted as wrong).

The rules were adjusted by hand looking only at the **tuning** photos; the **unseen** photos were never used for that, so their figures are the fair estimate for new menus.

| photos | count | reference items | found correctly | wrong | missed | repeats | recall | precision |
|---|---|---|---|---|---|---|---|---|
| all | 15 | 595 | 397 | 77 | 198 | 51 | 66.7% | 83.8% |
| tuning (dev) | 8 | 426 | 307 | 29 | 119 | 24 | 72.1% | 91.4% |
| unseen (test) | 7 | 169 | 90 | 48 | 79 | 27 | 53.3% | 65.2% |

- Photos: 19 (15 with reference answers, 0 without, 4 copies of another photo, not scored)
- LLM used on 26.3% of the photos (100.0% of them answered from the cache)
- Cost: $0.0000 total, $0.0000 per photo
- Time: 365.33s total, 19.23s per photo; LLM 288.09s total, 57.62s per LLM call
- LLM items discarded because they are not in the photo's text (likely invented): 36
- Photos with errors: 3
  - `1326306244314649407/8601bf03078cf8be78423f3b5835391a`: llm: Ollama connection error (ollama/qwen2.5:7b): timed out
  - `2564282930447335749/2f04957e9038d0282481700cbf990c80`: llm: Ollama connection error (ollama/qwen2.5:7b): timed out
  - `2564282930447335749/91b12c85a02f2d5014d2025de06e9c9e`: llm: no grounded dishes, kept rules output

## Per photo

| photo | set | items found | reference items | correct | wrong | missed | repeats | recall | precision | LLM used | cost | time | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `10320618957461533705/5b1f2c20f1b2db7837099aa9e9da6037` | tuning | 76 | 75 | 73 | 3 | 2 | 0 | 97.3% | 96.1% | no | $0.0000 | 13.03s |  |
| `10320618957461533705/7167d9d1e8001dbdcb304e4db4da9233` | tuning | 69 | 45 | 42 | 3 | 3 | 24 | 93.3% | 93.3% | no | $0.0000 | 4.02s |  |
| `12353399953336620223/2348ac983ccbea08aea4af7896f68c00` | tuning | 77 | 102 | 70 | 7 | 32 | 0 | 68.6% | 90.9% | no | $0.0000 | 4.57s |  |
| `12353399953336620223/a8c3a678c184ef488d25f9774c20128d` | tuning | 41 | 43 | 39 | 2 | 4 | 0 | 90.7% | 95.1% | no | $0.0000 | 1.98s |  |
| `1326306244314649407/80ada8490cd0992945a911e96597d8e4` | tuning | 1 | 18 | 1 | 0 | 17 | 0 | 5.6% | 100.0% | yes | $0.0000 | 40.24s |  |
| `1326306244314649407/8601bf03078cf8be78423f3b5835391a` | tuning | 25 | 53 | 18 | 7 | 35 | 0 | 34.0% | 72.0% | no | $0.0000 | 3.04s | yes |
| `13704801545575264378/594bd6c734095e01b002c63e7e48614b` | tuning | 62 | 78 | 59 | 3 | 19 | 0 | 75.6% | 95.2% | no | $0.0000 | 3.22s |  |
| `13704801545575264378/ff5e0547d14515a853d60a8cc9063dd3` | copy, not scored | 62 | 78 | 59 | 3 | 19 | 0 | 75.6% | 95.2% | no | $0.0000 | 0.00s |  |
| `16055038156589254689/50b409d7bee89d6fab5946ba766b2a4b` | tuning | 9 | 12 | 5 | 4 | 7 | 0 | 41.7% | 55.6% | yes | $0.0000 | 54.92s |  |
| `16055038156589254689/73f185f9fc81066b39c0e71d111c4d97` | copy, not scored | 9 | 12 | 5 | 4 | 7 | 0 | 41.7% | 55.6% | no | $0.0000 | 0.00s |  |
| `17453160375733808238/2ec92074a9e3b06948d12b09f5183925` | unseen | 10 | 10 | 10 | 0 | 0 | 0 | 100.0% | 100.0% | no | $0.0000 | 1.76s |  |
| `17453160375733808238/96e7cbb9c363bb62c949b804b1b6fd1a` | unseen | 49 | 18 | 13 | 11 | 5 | 25 | 72.2% | 54.2% | yes | $0.0000 | 104.10s |  |
| `2564282930447335749/2f04957e9038d0282481700cbf990c80` | unseen | 16 | 35 | 4 | 12 | 31 | 0 | 11.4% | 25.0% | no | $0.0000 | 15.50s | yes |
| `2564282930447335749/91b12c85a02f2d5014d2025de06e9c9e` | unseen | 30 | 45 | 9 | 21 | 36 | 0 | 20.0% | 30.0% | yes | $0.0000 | 105.85s | yes |
| `7571032721762097332/63bac2469809e9b1432e6f152996440d` | unseen | 16 | 18 | 15 | 1 | 3 | 0 | 83.3% | 93.8% | no | $0.0000 | 3.60s |  |
| `7571032721762097332/8a339445e1a8029008955dc43d14f078` | copy, not scored | 16 | 18 | 15 | 1 | 3 | 0 | 83.3% | 93.8% | no | $0.0000 | 0.00s |  |
| `8627417098818383545/1bbb57ae336f4728a1b312b7fb787461` | unseen | 41 | 41 | 38 | 1 | 3 | 2 | 92.7% | 97.4% | no | $0.0000 | 2.83s |  |
| `8627417098818383545/a1ae40f3a8b8acf06d0ba25b2157d928` | copy, not scored | 41 | 41 | 38 | 1 | 3 | 2 | 92.7% | 97.4% | no | $0.0000 | 0.00s |  |
| `9944003937257840785/3cbb7ebc3d0c8378482b5a96a94e5278` | unseen | 3 | 2 | 1 | 2 | 1 | 0 | 50.0% | 33.3% | yes | $0.0000 | 6.66s |  |

## What went wrong, per photo

### `10320618957461533705/5b1f2c20f1b2db7837099aa9e9da6037` (recall 97.3%)
- Missed: "Pollo Chili (Picante)", "Cordero Rogan Josh"
- Wrong: "salado es lo que hará especial a este plato", "Trozos de pollo rebozado con harina de garban", "cebollas y una variedad de especias auténticas"

### `10320618957461533705/7167d9d1e8001dbdcb304e4db4da9233` (recall 93.3%)
- Missed: "Samosas vegetal", "Dahi Puri", "Samosas de carne"
- Wrong: "carne(2 uds.)", "TERFLY", "Polto ezi"

### `12353399953336620223/2348ac983ccbea08aea4af7896f68c00` (recall 68.6%)
- Missed: "EDAMAME", "AGEDASHI TOFU", "YAKI GYOZA", "GYOZA", "HARGAO" (+27 more)
- Wrong: "ATUN", "Mak", "Mako de Saimón", "Maku de Adan", "Rollo grande" (+2 more)

### `12353399953336620223/a8c3a678c184ef488d25f9774c20128d` (recall 90.7%)
- Missed: "TEMAKI", "SUSHI VARIADO (TYU/ZYO/KOKUJO)", "Uramaki de salmón semicrudo y langostinos rebozados", "TONKASU"
- Wrong: "TEMAKL", "y langostinos rebozados"

### `1326306244314649407/80ada8490cd0992945a911e96597d8e4` (recall 5.6%)
- Missed: "Margarita", "4 quesos Plus", "Todo carne", "Classic Plus", "Hawaiana" (+12 more)

### `1326306244314649407/8601bf03078cf8be78423f3b5835391a` (recall 34.0%)
- Missed: "THE BOSTON", "TARTUFATA", "CABRA", "PAN DE AJO", "ALITAS BARBACOA / PICANTES" (+30 more)
- Wrong: "Añade los ingredientes", "HELADOS", "SALSAS (TOMATE / PICANTE / BBQ / PESTO)", "PAN AJO -2A PIZZA AL 50% IGUAL O MENOR QUE LA 1A", "INGREDIENTES" (+2 more)

### `13704801545575264378/594bd6c734095e01b002c63e7e48614b` (recall 75.6%)
- Missed: "Pani Puri", "Mysore Masala Dosa", "Idly", "Tandoori Baingan Bartha", "Daal Makhani" (+14 more)
- Wrong: "Streat Food Specialties", "OVOSG", "Tandoori"

### `13704801545575264378/ff5e0547d14515a853d60a8cc9063dd3` (recall 75.6%)
- Missed: "Pani Puri", "Mysore Masala Dosa", "Idly", "Tandoori Baingan Bartha", "Daal Makhani" (+14 more)
- Wrong: "Streat Food Specialties", "OVOSG", "Tandoori"

### `16055038156589254689/50b409d7bee89d6fab5946ba766b2a4b` (recall 41.7%)
- Missed: "LAMB KARAHI", "LAMB NILGIRI", "LAMB DAHI", "LAMB BOTI MASALA", "MANGO LAMB" (+2 more)
- Wrong: "XAI-CORDERO-LAMB", "FISH", "FISH", "JHIN"

### `16055038156589254689/73f185f9fc81066b39c0e71d111c4d97` (recall 41.7%)
- Missed: "LAMB KARAHI", "LAMB NILGIRI", "LAMB DAHI", "LAMB BOTI MASALA", "MANGO LAMB" (+2 more)
- Wrong: "XAI-CORDERO-LAMB", "FISH", "FISH", "JHIN"

### `17453160375733808238/96e7cbb9c363bb62c949b804b1b6fd1a` (recall 72.2%)
- Missed: "Sashimi de salmón", "Bola de arroz cubierto con loncha de pescado crudo", "Pulpo con trufa", "Aguacate", "Tortilla japonesa"
- Wrong: "2,50€/1uds", "10,95 €/5uds", "19.50 €/25uds", "Peix cru variat en rodanxes", "Variado" (+6 more)

### `2564282930447335749/2f04957e9038d0282481700cbf990c80` (recall 11.4%)
- Missed: "AMERICANA", "BBQ", "CARB-O-NADA", "CLOTENCA", "DELIZZIA" (+26 more)
- Wrong: "Amb brie i tòfona negra", "Amb formatge", "4 unitats", "DE TEMPORADA", "PER NOMÉS" (+7 more)

### `2564282930447335749/91b12c85a02f2d5014d2025de06e9c9e` (recall 20.0%)
- Missed: "SHEPPERD", "DELIGHT", "GORGON-Z", "DE LUX", "AL GUST" (+31 more)
- Wrong: "tomáquet a rodanxes, escalopa", "tomàquet a rodanxes, burger de", "Pa de xapata amb sèsam, enciam", "Pa de xapata amb sèsam, enciam", "Pa de xapata amb sèsam, enciam i" (+16 more)

### `7571032721762097332/63bac2469809e9b1432e6f152996440d` (recall 83.3%)
- Missed: "Verde di Búfala", "Nestea", "Lambrusco"
- Wrong: "cacerola"

### `7571032721762097332/8a339445e1a8029008955dc43d14f078` (recall 83.3%)
- Missed: "Verde di Búfala", "Nestea", "Lambrusco"
- Wrong: "cacerola"

### `8627417098818383545/1bbb57ae336f4728a1b312b7fb787461` (recall 92.7%)
- Missed: "Combinado Machawi (2 personas)", "Humus", "Pan"
- Wrong: "Empanadillas de carne, 2 de queso y 2 de espinacas"

### `8627417098818383545/a1ae40f3a8b8acf06d0ba25b2157d928` (recall 92.7%)
- Missed: "Combinado Machawi (2 personas)", "Humus", "Pan"
- Wrong: "Empanadillas de carne, 2 de queso y 2 de espinacas"

### `9944003937257840785/3cbb7ebc3d0c8378482b5a96a94e5278` (recall 50.0%)
- Missed: "TORTITA AMERICANA CASERA"
- Wrong: "J-TORTITA ANERICANA", "SIROPE"
