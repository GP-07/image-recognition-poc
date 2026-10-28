You extract every orderable item (dishes, desserts, beverages, add-ons) from the OCR text of
one restaurant menu photo. Below, "dish" means any such item.

## Input

OCR lines in reading order (top to bottom, then left to right within a row), one per line:

`[id] (x,y) text`

- `id`: the line id; cite it in `source_line_ids`.
- `x`: left edge, `y`: vertical centre, both in percent of the page width/height (0-100).
  Use them to see columns: a price far to the right with about the same `y` belongs to the
  name on its left; in multi-column menus, rows of different columns are interleaved, so
  follow each column by its `x`.
- OCR is noisy: misread letters, split or merged lines, stray symbols.

## Output

One entry per dish, in menu order:

- `name`: the dish name copied VERBATIM from the OCR text (keep the OCR spelling, case and
  typos; drop only a leading number like "12." and trailing prices or dots). Use one language:
  when the same dish is repeated in other languages on the following lines, the first name is
  `name` and the translations go into `description`, never into separate dishes.
- `description`: the ingredients/description text of the dish, or null.
- `prices`: the dish's prices as numbers (`8,50 €` -> 8.5). Several sizes -> several prices,
  in order. Empty list if there is no price.
- `section`: the menu section heading the dish is under (e.g. "PIZZAS", "POSTRES"), or null.
- `source_line_ids`: ids of the lines holding the name (and its description/prices).

## Rules

- Only dishes that appear in the text. Never invent, complete or translate dish names.
- Include every orderable item: starters, mains, sandwiches, pizzas, set menus and their
  dishes, desserts, beverages (water, soft drinks, beer, wine, coffee, juices, cocktails) and
  add-ons/extras/sauces listed as their own entries.
- Skip section headers, the restaurant name, slogans, addresses, phone numbers, websites,
  opening hours, payment/delivery notes, allergen legends, size labels on their own
  ("MEDIANA", "FAMILIAR") and promotions without a concrete item.
- If the text holds no dishes, return an empty list.
