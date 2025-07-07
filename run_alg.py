import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-db", help="path to dataset input file")
parser.add_argument("-dbl", help="path to labels input file")
parser.add_argument("-op", help="output prefix (def. empty)", default="")
parser.add_argument(
    "-r", help="regularization parameter (def 0.0001)", type=float, default=0.0001)
parser.add_argument(
    "-k", help="max number of rules in a rule list (def = 5)", type=int, default=5)
parser.add_argument("-v", help="verbose level (def. 1)", type=int, default=1)
args = parser.parse_args()

# Extract directory from output prefix to ensure corelsOut directory exists
prefix_dir = os.path.dirname(args.op)
corelsOut_dir = os.path.join(
    prefix_dir, 'corelsOut') if prefix_dir else 'corelsOut'
if not os.path.exists(corelsOut_dir):
    os.makedirs(corelsOut_dir, exist_ok=True)
    if args.v > 0:
        print(f"Created directory: {corelsOut_dir}")

max_num_nodes = "1000000000"
corels_cmd = f"./src/corels -n {max_num_nodes} -r {args.r} -c 1 -p 1 {args.db} {args.dbl} -d {args.k}"

# Use base filename, not full path for the output file
base_name = os.path.basename(args.op) if args.op else ""
fout_path = os.path.join(corelsOut_dir, f"{base_name}out.txt")

if args.v > 0:
    cmd = f"script -c \"{corels_cmd}\" -f {fout_path}"
    print(cmd)
else:
    cmd = f"{corels_cmd} > {fout_path}"
ret_val = os.system(cmd)
if ret_val != 0 and ret_val != 35584:  # 139 << 8 = 35584
    print("ERROR with CORELS!!!")
    print(cmd)
    exit(1)
