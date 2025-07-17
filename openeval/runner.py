import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from .test_utils import run_test
from tqdm import tqdm

def _temp_run(inputs, outputs, fn_name, generation, debug, result, timeout, mem_limit):
    res = run_test(inputs, outputs, fn_name, generation, debug, timeout, mem_limit)
    result.append(res)
    # metadata_list.append(metadata)


def check_correctness(inputs, outputs, fn_name, generation, timeout, mem_limit, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(inputs, outputs, fn_name, generation, debug, result, timeout, mem_limit),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(inputs) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        # consider that all tests failed
        result = [([-1], []), ]
        if debug:
            print(f"global timeout")
    return result[0]