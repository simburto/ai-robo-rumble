import numpy as np
from multiprocessing import *
from concurrent.futures.process import ProcessPoolExecutor
import time
from ctypes import c_double


NUM_WORKERS = cpu_count()
np.random.seed(42)
ARRAY_SIZE = int(2e8)
data = np.random.random(ARRAY_SIZE)
NP_DATA_TYPE = np.float64
ARRAY_SHAPE = (ARRAY_SIZE,)

shared_array = RawArray(c_double, ARRAY_SIZE)
shared_array_np = np.ndarray(ARRAY_SHAPE, dtype=NP_DATA_TYPE, buffer=shared_array)
# Copy data to our shared array.
np.copyto(shared_array_np, data)

def np_sum_shared_array(start, stop):
    return np.sum(shared_array_np[start:stop])

def benchmark():
    chunk_size = int(ARRAY_SIZE / NUM_WORKERS)
    futures = []
    ts = time.time_ns()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i in range(0, NUM_WORKERS):
            start = i + chunk_size if i == 0 else 0
            futures.append(executor.submit(np_sum_shared_array, start, i + chunk_size))
    futures, _ = concurrent.futures.wait(futures)
    return (time.time_ns() - ts) / 1_000_000