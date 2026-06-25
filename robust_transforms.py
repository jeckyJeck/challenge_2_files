"""
robust_transforms.py  —  owned by the evaluation-pipeline person.

A battery of "label-preserving" corruptions that simulate the kinds of
visual manipulations the hidden test set will use: background/color context
shifts, lighting changes, and color distortions (see challenge Section 2.2).

Pure PIL + numpy on purpose, so the eval side has NO torch dependency and
can be tested in isolation. Each transform maps PIL.Image -> PIL.Image at a
fixed, reproducible severity (eval must be deterministic).

KEY IDEA for honest OOD measurement
------------------------------------
The real test augmentations are HIDDEN. If your teammate trains on the exact
same corruptions you evaluate on, your local "robust accuracy" is optimistic.
So the families are tagged TRAIN or HELDOUT:
  - share the TRAIN families with the training person (they augment with these)
  - keep the HELDOUT families secret from training; evaluate on them to
    estimate generalisation to *unseen* manipulations.
This mirrors the grader's in-domain vs out-of-domain split.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

# ----------------------------------------------------------------------------- helpers
def _np(img):  return np.asarray(img.convert("RGB"), dtype=np.float32)
def _img(arr): return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


# ----------------------------------------------------------------------------- lighting
def brightness_down(img): return ImageEnhance.Brightness(img).enhance(0.55)
def brightness_up(img):   return ImageEnhance.Brightness(img).enhance(1.6)
def low_contrast(img):    return ImageEnhance.Contrast(img).enhance(0.5)
def gamma_dark(img):
    a = _np(img) / 255.0
    return _img((a ** 1.8) * 255.0)


# ----------------------------------------------------------------------------- colour / context
def hue_rotate(img):
    h, s, v = img.convert("HSV").split()
    h = h.point(lambda p: (p + 90) % 256)            # rotate hue ~126 deg
    return Image.merge("HSV", (h, s, v)).convert("RGB")

def desaturate(img):      return ImageEnhance.Color(img).enhance(0.25)
def oversaturate(img):    return ImageEnhance.Color(img).enhance(2.2)

def channel_swap(img):                                # R<->B : strong colour cast
    a = _np(img); a = a[:, :, ::-1]
    return _img(a)

def color_cast(img, rgb=(1.25, 1.0, 0.7)):            # warm tint, mimics lighting/bg shift
    a = _np(img) * np.array(rgb, dtype=np.float32)
    return _img(a)

def cool_cast(img):       return color_cast(img, (0.75, 0.95, 1.3))


# ----------------------------------------------------------------------------- distortion / corruption
def to_grayscale(img):    return ImageOps.grayscale(img).convert("RGB")
def invert(img):          return ImageOps.invert(img.convert("RGB"))
def posterize(img):       return ImageOps.posterize(img.convert("RGB"), 3)
def solarize(img):        return ImageOps.solarize(img.convert("RGB"), threshold=110)

def gaussian_noise(img, sigma=28.0):
    rng = np.random.default_rng(0)                   # fixed seed -> deterministic eval
    a = _np(img) + rng.normal(0, sigma, _np(img).shape)
    return _img(a)

def gaussian_blur(img):   return img.convert("RGB").filter(ImageFilter.GaussianBlur(2.2))


# ----------------------------------------------------------------------------- registries
# Tag each family. Share TRAIN with the training teammate; keep HELDOUT for eval only.
TRAIN_FAMILIES = {
    "brightness_down": brightness_down,
    "brightness_up":   brightness_up,
    "hue_rotate":      hue_rotate,
    "desaturate":      desaturate,
    "color_cast_warm": color_cast,
    "grayscale":       to_grayscale,
    "gaussian_noise":  gaussian_noise,
}

HELDOUT_FAMILIES = {                                  # NEVER train on these
    "low_contrast":    low_contrast,
    "gamma_dark":      gamma_dark,
    "channel_swap":    channel_swap,
    "cool_cast":       cool_cast,
    "oversaturate":    oversaturate,
    "invert":          invert,
    "posterize":       posterize,
    "solarize":        solarize,
    "gaussian_blur":   gaussian_blur,
}

ALL_FAMILIES = {**TRAIN_FAMILIES, **HELDOUT_FAMILIES}


def identity(img):  # clean reference
    return img.convert("RGB")