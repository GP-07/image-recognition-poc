import pytest

from menu_poc.evaluation.golden import GoldenError, expected_filename, load_golden

GOOD = """\
image_id: "r1/menu_a"
split: dev
reviewed: true
notes: "two languages"
dishes:
  - name: "Lamb Curry"
    aliases: ["Xai al curry", "Cordero al curry"]
  - name: "Naan"
"""


def test_expected_filename():
    assert expected_filename("r1/menu_a") == "r1__menu_a.yaml"


def test_load_golden_ok(tmp_path):
    (tmp_path / "r1__menu_a.yaml").write_text(GOOD)
    golden = load_golden(tmp_path)
    image = golden["r1/menu_a"]
    assert image.split == "dev" and image.reviewed
    assert image.dishes[0].variants == ["Lamb Curry", "Xai al curry", "Cordero al curry"]
    assert image.dishes[1].aliases == []


def test_only_reviewed_skips_unreviewed(tmp_path):
    (tmp_path / "r1__menu_a.yaml").write_text(GOOD.replace("reviewed: true", "reviewed: false"))
    assert load_golden(tmp_path) != {}
    assert load_golden(tmp_path, only_reviewed=True) == {}


def test_missing_dir_is_empty(tmp_path):
    assert load_golden(tmp_path / "nope") == {}


@pytest.mark.parametrize(
    "content",
    [
        GOOD.replace("split: dev", "split: train"),
        GOOD.replace('name: "Naan"', 'name: ""'),
        "image_id: [unclosed",
        "- just a list",
    ],
)
def test_invalid_file_reports_file_name(tmp_path, content):
    (tmp_path / "r1__menu_a.yaml").write_text(content)
    with pytest.raises(GoldenError, match="r1__menu_a.yaml"):
        load_golden(tmp_path)


def test_file_name_must_match_image_id(tmp_path):
    (tmp_path / "r1__other.yaml").write_text(GOOD)
    with pytest.raises(GoldenError, match="r1__other.yaml"):
        load_golden(tmp_path)
