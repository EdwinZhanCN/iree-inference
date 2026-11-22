"""Synthetic text images, preprocessed like PP-OCR (BGR, NCHW float32).

det: a page of text lines, /255 then ImageNet mean/std.
rec: one text line cropped to its text, resized to the model height keeping
its aspect ratio, normalized to x/127.5 - 1, zero-padded on the right
(PaddleOCR's RecResizeImg; esp-dl's pp_ocr_v6 does the same).

They give realistic activation ranges for calibration and comparison. They are
not an accuracy benchmark.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CALIB_SEED = 1234
TEST_SEED = 99
TEST_IMAGES = 3

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FONTS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
WORDS = (
    "IREE ESP32 P4 OCR invoice total 2026 serial No. 0x3F7A price 12.50 "
    "hello world receipt model"
).split()


def _font(size, rng):
    for path in rng.permutation(FONTS):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def text_image(h, w, rng):
    bg = tuple(int(v) for v in rng.integers(170, 256, 3))
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(3, 12))):
        size = int(rng.integers(max(10, h // 40), max(12, h // 8)))
        text = " ".join(rng.choice(WORDS, int(rng.integers(1, 5))))
        x = int(rng.integers(0, max(1, w // 2)))
        y = int(rng.integers(0, max(1, h - size)))
        fg = tuple(int(v) for v in rng.integers(0, 90, 3))
        draw.text((x, y), text, fill=fg, font=_font(size, rng))
    return img


def text_line(h, w, rng):
    """One text line as a crop would come out of det, resized to height h."""
    size = 64
    font = _font(size, rng)
    text = " ".join(rng.choice(WORDS, int(rng.integers(1, 5))))
    left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    pad_x, pad_y = int(rng.integers(2, 12)), int(rng.integers(4, 14))
    bg = tuple(int(v) for v in rng.integers(170, 256, 3))
    img = Image.new("RGB", (right - left + 2 * pad_x, bottom - top + 2 * pad_y), bg)
    fg = tuple(int(v) for v in rng.integers(0, 90, 3))
    ImageDraw.Draw(img).text((pad_x - left, pad_y - top), text, fill=fg, font=font)
    rw = max(1, min(w, int(np.ceil(h * img.width / img.height))))
    return img.resize((rw, h), Image.BILINEAR), text


def preprocess(img, kind, w=None):
    a = np.asarray(img, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
    if kind == "det":
        a = (a / 255.0 - MEAN) / STD
    else:
        a = a / 127.5 - 1.0
        a = np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)))
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def batch(kind, n, h, w, seed):
    """n inputs for `det` (a page of text) or `rec` (one text line)."""
    if kind == "det":
        rng = np.random.default_rng(seed)
        return [preprocess(text_image(h, w, rng), kind) for _ in range(n)]
    return rec_batch(n, h, w, seed)[0]


def rec_batch(n, h, w, seed):
    """n rec inputs and the text drawn on each."""
    rng = np.random.default_rng(seed)
    lines = [text_line(h, w, rng) for _ in range(n)]
    return [preprocess(img, "rec", w) for img, _ in lines], [t for _, t in lines]
