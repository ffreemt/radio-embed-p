"""Embed input."""
# pylint: disable=invalid-name
import multiprocessing as mp
from math import ceil

import more_itertools as mit
import numpy
from hf_model_s_cpu import model_s
from logzero import logger

try:
    model = model_s()
except Exception as _:
    logger.exception(_)
    raise SystemExit(1) from _

num_cpus = mp.cpu_count()


def radio_embed(
    text: str,
) -> numpy.ndarray:
    """Embed input."""
    try:
        _ = model.encode(text.strip().splitlines())
    except Exception as _:
        logger.exception(_)
        raise

    return _


def radio_embed_p(
    text: str,
) -> numpy.ndarray:
    """Embed input in parallel."""
    # split evenly to num_cpus parts
    _ = text.splitlines()
    _ = mit.chunked_even(_, ceil(len(_) / num_cpus))

    # back to str for radio_embed
    args = ["\n".join(elm) for elm in _]

    try:
        with mp.Pool(num_cpus) as pool:
            # ret = pool.map(func, args_)
            ret = pool.map(radio_embed, args)
    except Exception as exc:
        logger.error(exc)
        raise

    return ret
