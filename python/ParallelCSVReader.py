import pandas as pd
import numpy as np
import threading


class ParallelCSVReader(threading.Thread):
    def __init__(self, filename, begin, end):
        super().__init__()
        self.filename = filename
        self.begin = begin
        self.end = end
        self.loaded = False
        self.df = None

    def run(self):
        r = list(range(self.begin, self.end))
        r = [int(i) for i in r]
        cols = np.concatenate([[0], r])
        self.df = pd.read_csv(self.filename,
                              compression="gzip",
                              index_col=0,
                              dtype=np.float32,
                              usecols=cols
                              )
        self.loaded = True

    def get_df(self):
        return self.df


class ParallelCSVReaderLoader:
    def __init__(self, thread_num, filename):
        COL_MAX = self.get_column_count(filename)
        part_size = int(np.ceil(COL_MAX / thread_num))
        begins = [0] * thread_num
        ends = [0] * thread_num
        for i in range(thread_num):
            if i == 0:
                begins[i] = 1
                ends[i] = part_size + 1
            else:
                begins[i] = ends[i - 1]
                ends[i] = min(begins[i] + part_size + 1, COL_MAX)
        self.readers = []
        for i in range(thread_num):
            begins[i] = int(begins[i])
            ends[i] = int(ends[i])
            reader = ParallelCSVReader(filename, begins[i], ends[i])
            self.readers.append(reader)

        self.begins = begins
        self.ends = ends

    def get_column_count(self, filename):
        chunks = pd.read_csv(filename, chunksize=1)
        for c in chunks:
            return len(c.columns)

    def start(self):
        for reader in self.readers:
            reader.start()

    def wait(self):
        for reader in self.readers:
            reader.join()

    def concat_df(self):
        dfs = []
        for reader in self.readers:
            dfs.append(reader.get_df())
        return pd.concat(dfs, axis=1)
