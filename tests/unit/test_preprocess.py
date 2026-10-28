import io

from PIL import Image

from menu_poc.preprocess.image import ImagePreprocessor


def test_downscales_longest_side_and_keeps_aspect():
    out = ImagePreprocessor(max_side=100)(Image.new("RGB", (400, 200), "white"))
    assert out.size == (100, 50)


def test_small_images_are_untouched_and_converted_to_rgb():
    out = ImagePreprocessor(max_side=100)(Image.new("L", (40, 30), 255))
    assert out.size == (40, 30)
    assert out.mode == "RGB"


def test_applies_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90° clockwise to display
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, format="JPEG", exif=exif.tobytes())
    image = Image.open(io.BytesIO(buf.getvalue()))
    assert ImagePreprocessor()(image).size == (20, 40)


def test_fingerprint_includes_settings():
    fp = ImagePreprocessor(max_side=1500).fingerprint()
    assert fp["max_side"] == 1500
    assert {"name", "version"} <= fp.keys()
