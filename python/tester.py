import time
import pandas as pd
import numpy as np

from ParallelCSVReader import ParallelCSVReaderLoader


def new_one():
    start = time.time()
    loader = ParallelCSVReaderLoader(16, "../output/reduced_test_merged.csv.gz")

    loader.start()
    loader.wait()
    print(time.time() - start)
    X = loader.concat_df()
    print(time.time() - start)


def old_one():
    start = time.time()
    X = pd.read_csv("../output/reduced_test_merged.csv.gz",
                    compression="gzip",
                    index_col=0,
                    dtype=np.float32
                    )
    print(time.time() - start)


if __name__ == '__main__':
    new_one()
