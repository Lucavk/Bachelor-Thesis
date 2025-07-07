import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-db", help="path to tabular dataset input file")
parser.add_argument(
    "-r", help="regularization parameter (def 0.0001)", type=float, default=0.0001)
parser.add_argument(
    "-k", help="max number of rules in a rule list (def = 5)", type=int, default=5)
parser.add_argument(
    "-z", help="max number of conjunctions in each rule", type=int, default=1)
parser.add_argument("-minf", help="min freq of conjunctions",
                    type=float, default=0.)
parser.add_argument("-op", help="output prefix (def. empty)", default="")
parser.add_argument(
    "-s", help="sampling rate (def=-1 for no sampling)", type=float, default=-1)
parser.add_argument(
    "-m", help="sampling size (def=-1 for no sampling)", type=int, default=-1)
parser.add_argument("-v", help="verbose level (def. 1)", type=int, default=1)
parser.add_argument(
    "-f", help="1 = force creating new sample, 0 = use sample as is (def. 1)", type=int, default=1)
parser.add_argument("-seed", help="random seed (def. None)",
                    type=int, default=None)
args = parser.parse_args()

# Extract directory from output prefix to make sure it exists
prefix_dir = os.path.dirname(args.op)
if prefix_dir and not os.path.exists(prefix_dir):
    os.makedirs(prefix_dir, exist_ok=True)
    if args.v > 0:
        print(f"Created directory: {prefix_dir}")

# Create a 'data' subdirectory within the output directory if needed
data_dir = os.path.join(prefix_dir, 'data') if prefix_dir else 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    if args.v > 0:
        print(f"Created directory: {data_dir}")

# Create a 'corelsOut' subdirectory for outputs if needed
corelsOut_dir = os.path.join(
    prefix_dir, 'corelsOut') if prefix_dir else 'corelsOut'
if not os.path.exists(corelsOut_dir):
    os.makedirs(corelsOut_dir, exist_ok=True)
    if args.v > 0:
        print(f"Created directory: {corelsOut_dir}")

# Use the correct paths for intermediate files
base_name = os.path.basename(args.op) if args.op else ""
sample_db = os.path.join(data_dir, f"{base_name}sample_db_cor.db")
sample_labels = os.path.join(data_dir, f"{base_name}sample_labels_cor.labels")

cmd = f"python3 tabularbinary_to_corels.py -db {args.db} -od {sample_db} -ol {sample_labels} -s {args.s} -m {args.m} -v {args.v} -z {args.z} -minf {args.minf}"
if args.seed is not None:
    cmd += f" -seed {args.seed}"
if args.v > 0:
    print(cmd)
    print()
if args.f > 0:
    ret_val = os.system(cmd)
    if ret_val > 0:
        print("ERROR with tabularbinary_to_corels.py!!!")
        print(cmd)
        exit(1)

# Run the algorithm
cmd = f"python3 run_alg.py -db {sample_db} -dbl {sample_labels} -k {args.k} -r {args.r} -op {args.op} -v {args.v}"
if args.v > 0:
    print(cmd)
    print()
ret_val = os.system(cmd)
if ret_val > 0:
    print("ERROR with run_alg.py!!!")
    print(cmd)
    exit(1)

# Remove the .db and .labels files
if os.path.exists(sample_db):
    os.remove(sample_db)
    if args.v > 2:
        print(f"Removed file: {sample_db}")

if os.path.exists(sample_labels):
    os.remove(sample_labels)
    if args.v > 2:
        print(f"Removed file: {sample_labels}")
