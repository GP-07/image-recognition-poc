# Golden labels (ground truth) for menu → dishes extraction

> **Status: DRAFT, NOT REVIEWED.** These labels were drafted by an AI model (Claude, Anthropic)
> by looking at the 19 menu photos. Every file has `reviewed: false`. A human must review them
> before any metric computed against them is trusted.

## Known bias risk

The labeler (Claude) belongs to the same model family as the LLM being evaluated in this POC.
The two may share reading errors, normalisation habits (e.g. how a name is split from its
description) and blind spots, which would inflate measured precision/recall. Human review is the
mitigation; reviewers should pay special attention to the items flagged in each file's `notes`.

## Format

One YAML file per image: `data/golden/<restaurant_id>__<file_stem>.yaml`

```yaml
image_id: "16055038156589254689/73f185f9fc81066b39c0e71d111c4d97"
split: dev            # dev | test, assigned per restaurant
reviewed: false       # set to true once a human has checked the file
notes: "short description of the photo and anything uncertain / left out"
dishes:
  - name: "LAMB CURRY"
    aliases: ["Xai al curry tradicional", "Cordero al curry tradicional"]
```

## Labeling rules

- A dish = a food item a customer can order, as printed on the menu: starters, mains, pizzas,
  desserts, set menus (one entry per set menu; its component lines are not listed separately).
- Excluded: beverages (water, soft drinks, beer, wine, coffee, tea, juices, lassi/ayran),
  section headers, topping/extra lists, sauces sold as add-ons, allergen legends, prices, phone
  numbers, promotions without a concrete dish.
- `name`: as printed (menu language, spelling and casing kept, typos included); leading item
  numbers (`46.`) and quantity suffixes (`/ 4 PZS`, `(2 uds.)`) dropped.
- `aliases`: only the same dish name printed in another language (e.g. Catalan/Spanish/English
  lines that are direct translations of the name). Descriptions/ingredient lists and marketing
  taglines (e.g. La Clotenca's "O SOLE MIO") are NOT aliases.
- Only legible dishes are labeled. Cut-off, blurry or glare-covered items are left out and
  mentioned in `notes`. No guessing; two glare-damaged-but-reconstructible names are included
  and flagged (CALABAZA, PESTO in La Clotenca).
- When the same dish appears several times in one photo (e.g. on two boards), it is listed once.
- Exact or near-exact duplicate photos get identical labels (noted in `notes`).

## Split assignment (by restaurant)

| restaurant_id | split | menu / layout | difficulty |
|---|---|---|---|
| 10320618957461533705 | dev | Indian: rotated tri-fold menu; street stand with several menus | hard |
| 12353399953336620223 | dev | Japanese: night window photo with 4 menus; low-res single page | hard |
| 1326306244314649407 | dev | Pizzeria: poster + flat printed sheet | easy |
| 13704801545575264378 | dev | Veg World: trilingual vertical board, dense | medium |
| 16055038156589254689 | dev | Indian booklet page (lamb), trilingual | easy |
| 17453160375733808238 | test | Japanese photo-menu booklet, landscape | easy |
| 2564282930447335749 | test | La Clotenca: rotated paper menu, two sides, playful layout | medium |
| 7571032721762097332 | test | Vitali Pizza: dim sheet, salads + desserts + drinks | easy |
| 8627417098818383545 | test | Liban: street poster with photos, bilingual | medium |
| 9944003937257840785 | test | Handwritten promo sign over a plate of pancakes | edge case |

## Per-image summary

| image_id | split | #dishes | confidence | main caveats |
|---|---|---|---|---|
| 10320618957461533705/5b1f2c20f1b2db7837099aa9e9da6037 | dev | 75 | medium | rotated photo, many near-identical curry variants; verify count and spellings |
| 10320618957461533705/7167d9d1e8001dbdcb304e4db4da9233 | dev | 45 | medium | 3 overlapping menus; 2 Taj-Bowls items cut off and excluded |
| 12353399953336620223/2348ac983ccbea08aea4af7896f68c00 | dev | 100 | low | night/reflections; whole "Menu del dia" sheet excluded as unreadable; night-menu wording uncertain |
| 12353399953336620223/a8c3a678c184ef488d25f9774c20128d | dev | 43 | medium | low resolution (620x1385) |
| 1326306244314649407/80ada8490cd0992945a911e96597d8e4 | dev | 18 | high | – |
| 1326306244314649407/8601bf03078cf8be78423f3b5835391a | dev | 23 | high | "Festin" combos excluded as promotions |
| 13704801545575264378/594bd6c734095e01b002c63e7e48614b | dev | 59 | medium | alias-vs-description judgment on trilingual sub-lines |
| 13704801545575264378/ff5e0547d14515a853d60a8cc9063dd3 | dev | 59 | medium | pixel-identical duplicate of the above |
| 16055038156589254689/50b409d7bee89d6fab5946ba766b2a4b | dev | 12 | high | right page (items 61-77) cut off, excluded |
| 16055038156589254689/73f185f9fc81066b39c0e71d111c4d97 | dev | 12 | high | same photo at half resolution |
| 17453160375733808238/2ec92074a9e3b06948d12b09f5183925 | test | 10 | high | – |
| 17453160375733808238/96e7cbb9c363bb62c949b804b1b6fd1a | test | 18 | high | item 21 has only a descriptive name |
| 2564282930447335749/2f04957e9038d0282481700cbf990c80 | test | 31 | medium | taglines not used as aliases; CALABAZA reconstructed under glare |
| 2564282930447335749/91b12c85a02f2d5014d2025de06e9c9e | test | 22 | medium | pasta "combo": sauces labeled as dishes; PESTO reconstructed under glare |
| 7571032721762097332/63bac2469809e9b1432e6f152996440d | test | 10 | high | – |
| 7571032721762097332/8a339445e1a8029008955dc43d14f078 | test | 10 | high | pixel-identical duplicate of the above |
| 8627417098818383545/1bbb57ae336f4728a1b312b7fb787461 | test | 36 | high | Falafel (plate + sandwich) listed once; Ayran excluded as a drink |
| 8627417098818383545/a1ae40f3a8b8acf06d0ba25b2157d928 | test | 36 | high | same photo at half resolution |
| 9944003937257840785/3cbb7ebc3d0c8378482b5a96a94e5278 | test | 1 | medium | handwritten promo; name boundary ambiguous |

Totals: 19 images, 620 dish entries (dev: 10 images, 446 dishes; test: 9 images, 174 dishes).
Note: four restaurants contribute duplicate photos (13704801545575264378, 7571032721762097332
pixel-identical; 16055038156589254689 and 8627417098818383545 same photo at two resolutions),
so per-split metrics double-weight those menus.
