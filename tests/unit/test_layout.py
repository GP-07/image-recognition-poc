from menu_poc.models import OcrLine
from menu_poc.parsing.layout import (
    find_gutters,
    group_columns,
    group_rows,
    left_neighbor,
    reading_order,
    v_overlap,
)


def _line(line_id, x0, y0, x1, y1, text="text"):
    return OcrLine(id=line_id, text=text, bbox=(x0, y0, x1, y1), confidence=0.99)


def _two_columns():
    # Left column x 0..300, right column x 500..800, three rows each.
    left = [_line(row, 0, 100 + 40 * row, 300, 120 + 40 * row, f"L{row}") for row in range(3)]
    right = [
        _line(10 + row, 500, 100 + 40 * row, 800, 120 + 40 * row, f"R{row}") for row in range(3)
    ]
    return left + right


def test_v_overlap_is_relative_to_the_shorter_box():
    box = _line(0, 0, 0, 10, 20)
    assert v_overlap(box, _line(1, 0, 10, 10, 20)) == 1.0
    assert v_overlap(box, _line(2, 0, 30, 10, 40)) == 0.0


def test_group_rows_by_vertical_overlap():
    lines = _two_columns()
    rows = group_rows(lines)
    assert len(rows) == 3
    assert [ln.text for ln in rows[0]] == ["L0", "R0"]


def test_gutter_and_columns():
    lines = _two_columns()
    gutters = find_gutters(lines, width=900)
    assert len(gutters) == 1 and 300 <= gutters[0][0] < gutters[0][1] <= 500
    columns = group_columns(lines, width=900)
    assert [[ln.text for ln in col] for col in columns] == [["L0", "L1", "L2"], ["R0", "R1", "R2"]]


def test_reading_order_is_column_by_column():
    order = [ln.text for ln in reading_order(_two_columns(), width=900)]
    assert order == ["L0", "L1", "L2", "R0", "R1", "R2"]


def test_wide_title_goes_to_the_column_of_its_left_edge():
    lines = [*_two_columns(), _line(99, 0, 40, 800, 70, "TITLE")]
    columns = group_columns(lines, width=900)
    assert columns[0][0].text == "TITLE"


def test_left_neighbor_finds_name_of_a_far_price():
    name = _line(0, 0, 100, 200, 120, "Pollo Tikka")
    desc = _line(1, 20, 125, 300, 145, "con yogur")
    price = _line(2, 700, 102, 760, 122, "8,50€")
    assert left_neighbor(price, [name, desc]) == name


def test_left_neighbor_prefers_the_best_row_overlap():
    upper = _line(0, 0, 100, 200, 120, "upper")
    lower = _line(1, 0, 112, 350, 132, "lower")
    price = _line(2, 700, 104, 760, 124, "8,50€")
    assert left_neighbor(price, [upper, lower]) == upper


def test_left_neighbor_is_blocked_by_text_in_between():
    name = _line(0, 0, 100, 200, 120, "left column dish")
    other = _line(1, 400, 100, 600, 120, "right column dish")
    price = _line(2, 700, 100, 760, 120, "8,50€")
    assert left_neighbor(price, [name], obstacles=[name, other]) is None
    assert left_neighbor(price, [name, other]) == other


def test_left_neighbor_none_when_nothing_on_the_row():
    price = _line(0, 700, 100, 760, 120, "8,50€")
    assert left_neighbor(price, [_line(1, 0, 300, 200, 320)]) is None
