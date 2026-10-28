from menu_poc.models import OcrLine, OcrResult
from menu_poc.parsing.rules import RuleBasedParser, clean_name, is_noise

LINE_HEIGHT = 20  # line height of synthetic text
CHAR_W = 10


def _page(items, width=1000, height=1400):
    """``items``: (text, x0, y0) or (text, x0, y0, height)."""
    lines = []
    for line_id, item in enumerate(items):
        text, x0, y0 = item[:3]
        height_px = item[3] if len(item) > 3 else LINE_HEIGHT
        bbox = (x0, y0, x0 + CHAR_W * len(text), y0 + height_px)
        lines.append(OcrLine(id=line_id, text=text, bbox=bbox, confidence=0.97))
    return OcrResult(image_id="r/img", width=width, height=height, lines=lines, engine="fake")


def _by_name(result):
    return {dish.name: dish for dish in result.dishes}


def _parse(items, **kw):
    return RuleBasedParser().parse(_page(items, **kw))


def test_price_on_the_line_and_in_a_separate_column():
    result = _parse(
        [
            ("ENTRANTES", 50, 100, 30),
            ("Hummus ....... 5,50€", 50, 150),
            ("Crema de garbanzos", 70, 172),
            ("Falafel", 50, 210),
            ("6,00€", 600, 212),
            ("Croquetas vegetales", 70, 232),
            ("Tabule", 50, 270),
            ("4,90 €", 600, 268),
        ]
    )
    dishes = _by_name(result)
    assert set(dishes) == {"Hummus", "Falafel", "Tabule"}
    assert dishes["Hummus"].prices == [5.5]
    assert dishes["Hummus"].description == "Crema de garbanzos"
    assert dishes["Falafel"].prices == [6.0]
    assert dishes["Falafel"].description == "Croquetas vegetales"
    assert all(dish.section == "ENTRANTES" and dish.origin == "rules" for dish in result.dishes)
    assert dishes["Falafel"].source_line_ids == [3, 4, 5]
    assert result.confidence >= 0.6


def test_several_sizes_on_one_row():
    result = _parse([("Margarita", 50, 100), ("8,15€", 500, 100), ("9,50€", 600, 100)])
    assert result.dishes[0].prices == [8.15, 9.5]


def test_numbered_name_takes_price_from_its_translation():
    result = _parse(
        [
            ("46. LAMB CURRY", 50, 100),
            ("Xai al curry tradicional", 70, 122),
            ("Cordero al curry tradicional 15,90€", 70, 144),
            ("47. ROGAN GOSHT", 50, 190),
            ("Cordero en salsa 15,90€", 70, 212),
        ]
    )
    names = [dish.name for dish in result.dishes]
    assert names == ["LAMB CURRY", "ROGAN GOSHT"]
    assert result.dishes[0].prices == [15.9]
    assert "Cordero al curry tradicional" in result.dishes[0].description


def test_price_below_the_description_goes_to_the_dish():
    result = _parse(
        [
            ("301. Philadelphia maki", 50, 100),
            ("Salmon, aguacate y queso", 70, 122),
            ("11,95 €/8uds", 70, 144),
            ("302. Tuna roll", 50, 200),
            ("Atun y pepino", 70, 222),
            ("12,95 €/8uds", 70, 244),
        ]
    )
    assert [(dish.name, dish.prices) for dish in result.dishes] == [
        ("Philadelphia maki", [11.95]),
        ("Tuna roll", [12.95]),
    ]


def test_caps_title_followed_by_priced_ingredients_is_a_dish():
    result = _parse(
        [
            ("BURGERS", 50, 60, 30),
            ("SHEPPERD", 50, 100),
            ("Pa de xapata amb sesam,", 50, 122),
            ("tomaquet, escalopa de seitan 6,90€", 50, 144),
        ]
    )
    (dish,) = result.dishes
    assert dish.name == "SHEPPERD"
    assert dish.prices == [6.9]
    assert dish.section == "BURGERS"


def test_sections_change_with_headers():
    result = _parse(
        [
            ("POSTRES", 50, 100, 30),
            ("Tiramisu 4,50€", 50, 140),
            ("Ensaladas", 50, 190, 40),
            ("Caprese 7,50€", 50, 240),
        ]
    )
    assert {dish.name: dish.section for dish in result.dishes} == {
        "Tiramisu": "POSTRES",
        "Caprese": "Ensaladas",
    }


def test_drinks_are_menu_items_too():
    result = _parse(
        [
            ("Tiramisu 4,50€", 50, 100),
            ("LATA 33 CL 2,20€", 50, 150),
            ("REFRESCOS 2,50€", 50, 200),
            ("Bebidas", 50, 260, 40),
            ("Agua 1,50€", 50, 310),
            ("Café 1,20€", 50, 350),
        ]
    )
    names = [dish.name for dish in result.dishes]
    assert {"Tiramisu", "Agua", "Café"} <= set(names)


def test_ordering_note_with_price_is_not_a_dish():
    # "(2p) por encargo" carries the price of the name on its row.
    result = _parse([("UNI NIGIRI", 50, 100), ("8,00 (2p) por encargo", 400, 100)])
    assert [(dish.name, dish.prices) for dish in result.dishes] == [("UNI NIGIRI", [8.0])]


def test_diet_codes_are_noise():
    assert all(is_noise(_line(code)) for code in ("O.V.F.S", "O.V.O.S.a.", "KSG"))
    assert not is_noise(_line("Naan"))


def test_bulleted_names_are_dishes_even_without_price():
    result = _parse(
        [
            ("· MARGARITA", 50, 100),
            ("Tomate y mozzarella", 70, 122),
            ("· HAWAIANA", 50, 160),
            ("Jamón dulce y piña", 70, 182),
            ("·PLATOS SECOS·", 50, 240, 40),
        ]
    )
    assert [dish.name for dish in result.dishes] == ["MARGARITA", "HAWAIANA"]
    assert result.dishes[0].description == "Tomate y mozzarella"


def test_price_on_a_description_tail_goes_to_the_next_name():
    # Tilted photo: the next dish's price sits on the row of the description's last line.
    result = _parse(
        [
            ("7. Murgh Makhni", 50, 100),
            ("11,50€", 600, 100),
            ("Es un plato del Punjab. La salsa esta hecha", 70, 122),
            ("a base de mantequilla y especias 10,00€", 70, 144),
            ("Pollo Mango", 50, 168),
        ]
    )
    dishes = _by_name(result)
    assert set(dishes) == {"Murgh Makhni", "Pollo Mango"}
    assert dishes["Murgh Makhni"].prices == [11.5]
    assert dishes["Pollo Mango"].prices == [10.0]


def test_caps_line_like_the_dish_name_starts_a_new_dish():
    # The second price was cut by the photo edge; its leader dots and the name style still tell
    # a new dish (a title would not end in leaders).
    result = _parse(
        [
            ("EBI TEMPURA", 50, 100),
            ("13,50", 600, 100),
            ("Langostinos rebozados", 50, 122),
            ("TORI KAISU..", 50, 160),
            ("Pollo frito con sesamo", 50, 182),
        ]
    )
    assert [dish.name for dish in result.dishes] == ["EBI TEMPURA", "TORI KAISU"]
    assert result.dishes[1].prices == []
    assert result.dishes[1].description == "Pollo frito con sesamo"


def test_repeated_logo_text_is_not_a_section():
    items = []
    top = 100
    for section in ("SOPAS", "CURRY", "TANDOORI"):
        items += [(section, 50, top, 30), ("TASTE OF INDIA", 50, top + 32)]
        items += [(f"Plato {section.lower()} 9,50€", 50, top + 60)]
        top += 120
    sections = {dish.section for dish in _parse(items).dishes}
    assert sections == {"SOPAS", "CURRY", "TANDOORI"}


def test_noise_lines_are_not_dishes():
    result = _parse(
        [
            ("Pizza Margarita 9,50€", 50, 100),
            ("Tel. 93 444 47 37", 50, 160),
            ("www.vitalipizza.com", 50, 200),
            ("IVA incluido. Minimo 12,50€", 50, 240),
            ("Abierto de 19:30 a 23:30", 50, 280),
            ("Consulte nuestra carta de alérgenos", 50, 320),
            ("info@restaurante.es", 50, 360),
        ]
    )
    assert [dish.name for dish in result.dishes] == ["Pizza Margarita"]


def _line(text, conf=0.95):
    return OcrLine(id=0, text=text, bbox=(0, 0, 100, 20), confidence=conf)


def test_is_noise_and_clean_name():
    assert is_noise(_line("@#"))
    assert is_noise(_line("Pollo curry", conf=0.3))
    assert not is_noise(_line("Pollo curry"))
    assert clean_name("46. LAMB CURRY ...... 15,90€") == ("LAMB CURRY", None)
    assert clean_name("1 Boston: cebolla, bacon") == ("Boston", "cebolla, bacon")
    assert clean_name(".Pollo Chili (Picante)") == ("Pollo Chili (Picante)", None)


def test_size_table_lowers_confidence():
    # Pizza names in one block, a separate grid of size labels with prices: not understood.
    items = [(f"PIZZA {name}", 50, 100 + 40 * row) for row, name in enumerate("ABCDEF")]
    items += [
        (f"{size} {price}€", 500, 400 + 30 * row)
        for row, (size, price) in enumerate(
            [("PEQUEÑA", "9,00"), ("MEDIANA", "12,00"), ("GRANDE", "15,00")] * 3
        )
    ]
    result = _parse(items)
    assert result.signals["price_attach"] < 0.5
    assert result.confidence < 0.6


def test_empty_and_priceless_pages():
    assert _parse([]).confidence == 0.0
    result = _parse([("BIENVENIDOS", 50, 100, 40), ("Gracias por su visita", 50, 200)])
    assert result.dishes == [] and result.confidence == 0.0
