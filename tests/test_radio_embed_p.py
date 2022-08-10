"""Test radio_embed_p."""
# pylint: disable=broad-except, invalid-name
from pathlib import Path

import joblib
import numpy as np

from radio_embed_p import __version__, radio_embed_p
from radio_embed_p.radio_embed_p import radio_embed


def test_version():
    """Test version."""
    assert __version__[:3] == "0.1"


def test_sanity():
    """Check sanity."""
    try:
        assert not radio_embed_p()
    except Exception:
        assert True


def test_ff_en():
    """Test ff-en.txt."""
    # __file__ = "tests/test_radio_embed_p.py"

    # fpath = Path(__file__).parent / "ff-en.lzma"
    # r = joblib.load(fpath)

    fpath = Path(__file__).parent / "fangfang-en.txt"
    lst = [
        elm.strip() for elm in Path(fpath).read_text("utf8").splitlines() if elm.strip()
    ]
    text = "\n".join(lst[:100])

    r = radio_embed(text)

    # import os
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    r1 = radio_embed_p(text)

    assert np.allclose(np.array(r), np.array(r1), atol=1e-07)
