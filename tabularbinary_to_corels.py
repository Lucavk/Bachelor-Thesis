import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import os
from time import time
from itertools import chain, combinations

parser = argparse.ArgumentParser()
parser.add_argument("-db", help="path to input file")
parser.add_argument("-od", help="path to output data file")
parser.add_argument("-ol", help="path to output labels file")
parser.add_argument(
    "-z", help="max number of conjunctions in each rule", type=int, default=1)
parser.add_argument("-minf", help="min freq of conjunctions",
                    type=float, default=0.)
parser.add_argument(
    "-s", help="sampling rate (def=-1 for no sampling)", type=float, default=-1)
parser.add_argument(
    "-m", help="sampling size (def=-1 for no sampling)", type=int, default=-1)
parser.add_argument("-v", help="verbose level (def. 1)", type=int, default=1)
parser.add_argument("-seed", help="random seed (def. None)",
                    type=int, default=None)
args = parser.parse_args()

# Create output directories if they don't exist
for output_path in [args.od, args.ol]:
    if output_path:
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            if args.v > 0:
                print(f"Created directory: {directory}")

# Determine correct separator based on file extension
if args.db.endswith('.csv'):
    with open(args.db, 'r') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line else ','
else:
    sep = ','

# Load the dataset
df = pd.read_csv(args.db, sep=sep)
df = df.astype(int)
if args.v > 0:
    print(df)

# Memory-efficient sampling function


def memory_efficient_sampling(df, sample_size, random_state=None):
    """Sample from dataframe in a memory-efficient way using chunks"""
    if args.v > 0:
        print(f"Using memory-efficient sampling: size={sample_size}")

    # If sample size is too large, use chunking
    if sample_size > 5000:
        if args.v > 0:
            print(
                f"Large sample size detected ({sample_size}), using chunked sampling")

        # Process in smaller chunks to avoid memory issues
        chunk_size = 1000
        samples = []
        remaining = sample_size
        chunk_seed = random_state

        while remaining > 0:
            # Take a small chunk
            size = min(chunk_size, remaining)
            if args.v > 1:
                print(
                    f"  Sampling chunk of {size} rows (remaining: {remaining})")

            # Use a derived seed for each chunk for reproducibility
            if chunk_seed is not None:
                chunk_seed = chunk_seed + 1

            chunk = df.sample(n=size, replace=True, random_state=chunk_seed)
            samples.append(chunk)
            remaining -= size

        result = pd.concat(samples)
        result.reset_index(drop=True, inplace=True)
        return result
    else:
        # For small samples, use standard approach
        return df.sample(n=sample_size, replace=True, random_state=random_state)


# sampling of data
if args.s > 0 or args.m > 0:
    if args.s > 0:
        # Use data replication for fractional sampling
        df = df.loc[df.index.repeat(args.s)]
    else:
        # Use memory-efficient sampling for sample size
        df = memory_efficient_sampling(df, args.m, random_state=args.seed)

    df.reset_index(drop=True, inplace=True)
    if args.v > 0:
        print("After resampling:")
        print(df)


def get_str_array(array_vals):
    # print(array_vals.shape[0])
    thrs = 3*(array_vals.shape[0]+10)
    vals_str = np.array2string(
        array_vals, separator=" ", prefix="", suffix="", threshold=thrs, max_line_width=thrs)
    return vals_str[1:-1]


# write label file
start = time()
if args.v > 0:
    print("Printing labels...")
target_col_name = "{T}"
df["{T=1}"] = df[target_col_name]
df["{T=0}"] = (df[target_col_name]-1)*(-1)
df_targ = df[["{T=1}", "{T=0}"]]
df_targ = df_targ.transpose(copy=True)
df_targ.to_csv(args.ol, sep=" ", header=None)
elaps = time()-start
if args.v > 0:
    print("Done in", elaps)

# write data file
start = time()
if args.v > 0:
    print("Printing data...")
df.drop(target_col_name, axis=1, inplace=True)
df.drop("{T=1}", axis=1, inplace=True)
df.drop("{T=0}", axis=1, inplace=True)
df = df.transpose(copy=True)
df.to_csv(args.od, sep=" ", header=None)
elaps = time()-start
if args.v > 0:
    print("Done in", elaps)
