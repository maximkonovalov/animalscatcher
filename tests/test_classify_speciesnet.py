"""Tests for ac._classify_speciesnet -- the adapter that normalizes
SpeciesNet's classifier API to the plain (label, confidence) pair
_process_frame expects, matching what _classify_dfne already does for
PytorchWildlife's own classifiers.

Uses a fake object shaped like speciesnet.SpeciesNetClassifier rather
than the real package: speciesnet is a large, optional dependency (see
requirements-speciesnet.txt) not installed for the default DFNE-only
test run, and this adapter's own logic (label parsing, BGR->RGB
conversion) doesn't need a real model loaded to verify."""
import numpy as np

import ac


class FakeSpeciesNetClassifier:
    """Mimics speciesnet.SpeciesNetClassifier's preprocess()/predict()
    shape closely enough to exercise ac._classify_speciesnet, and
    records what it was actually called with so tests can assert on
    color conversion happening before preprocess()."""

    def __init__(self, classes, scores):
        self._classes = classes
        self._scores = scores
        self.last_preprocess_arr = None

    def preprocess(self, pil_img, bboxes=None):
        self.last_preprocess_arr = np.array(pil_img)
        return "preprocessed"

    def predict(self, filepath, img):
        return {
            "filepath": filepath,
            "classifications": {"classes": self._classes,
                               "scores": self._scores},
        }


def bgr_crop():
    # A crop where each channel has a distinct, recognizable value, so
    # a BGR<->RGB mixup would show up as swapped channel values below.
    crop = np.zeros((10, 10, 3), dtype="uint8")
    crop[:, :, 0] = 10   # B
    crop[:, :, 1] = 20   # G
    crop[:, :, 2] = 30   # R
    return crop


def test_top_prediction_returns_common_name_and_score():
    classifier = FakeSpeciesNetClassifier(
        classes=["uuid;mammalia;carnivora;felidae;puma;concolor;mountain lion"],
        scores=[0.87])

    label, conf = ac._classify_speciesnet(classifier, bgr_crop())

    assert label == "Mountain Lion"
    assert conf == 0.87


def test_crop_is_converted_from_bgr_to_rgb_before_preprocess():
    classifier = FakeSpeciesNetClassifier(classes=["uuid;;;;;;coyote"],
                                          scores=[0.5])

    ac._classify_speciesnet(classifier, bgr_crop())

    # bgr_crop() has B=10, G=20, R=30; after BGR->RGB conversion the
    # array's channel order should be R=30, G=20, B=10.
    r, g, b = classifier.last_preprocess_arr[0, 0]
    assert (r, g, b) == (30, 20, 10)


def test_higher_taxa_prediction_with_fewer_fields_falls_back_gracefully():
    # A less-confident prediction may only resolve to a higher taxon,
    # leaving the trailing common-name field empty.
    classifier = FakeSpeciesNetClassifier(classes=["uuid;mammalia;;;;;"],
                                          scores=[0.3])

    label, conf = ac._classify_speciesnet(classifier, bgr_crop())

    assert label == "Mammalia"
    assert conf == 0.3


def test_no_classifications_returns_unknown():
    classifier = FakeSpeciesNetClassifier(classes=[], scores=[])

    label, conf = ac._classify_speciesnet(classifier, bgr_crop())

    assert label == "Unknown"
    assert conf == 0.0
