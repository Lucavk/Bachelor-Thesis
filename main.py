import argparse
import pandas as pd
import os
import math

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
parser.add_argument(
    "-exact", help="if > 0, run on the entire dataset with given duplication factor (default=0)", type=int, default=0)
parser.add_argument("-delta", help="confidence parameter",
                    type=float, default=0.05)
parser.add_argument(
    "-epsilon", help="relative accuracy parameter", type=float, default=1.)
parser.add_argument(
    "-theta", help="absolute accuracy parameter", type=float, default=0.01)
parser.add_argument("-op", help="output prefix (def. empty)", default="")
parser.add_argument(
    "-ores", help="output results path (def. results.csv)", default="results.csv")
parser.add_argument("-v", help="verbose level (def. 1)", type=int, default=0)
parser.add_argument(
    "-f", help="1 = force creating new sample, 0 = use sample as is (def. 1)", type=int, default=1)
parser.add_argument("-seed", help="random seed (def. None)",
                    type=int, default=None)
args = parser.parse_args()


prefix_dir = os.path.dirname(args.op)
if prefix_dir and not os.path.exists(prefix_dir):
    os.makedirs(prefix_dir, exist_ok=True)
    if args.v > 0:
        print(f"Created directory: {prefix_dir}")


corelsOut_dir = os.path.join(
    prefix_dir, 'corelsOut') if prefix_dir else 'corelsOut'
if not os.path.exists(corelsOut_dir):
    os.makedirs(corelsOut_dir, exist_ok=True)


results_dir = os.path.dirname(args.ores)
if results_dir and not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)

if args.exact == 0:
    eps = args.epsilon
    theta = args.theta
    delta = args.delta
else:
    eps = theta = delta = 1.0
k = args.k
z = args.z


def check_current_sample_size(m):
    omega = k*z*math.log(2*math.e*d/z)+2
    ln_m = math.log(2./delta)/m
    ln_w_m = (omega+math.log(2./delta))/m
    term1 = math.sqrt(3*theta*ln_m)
    term2 = math.sqrt(2*(theta+term1)*ln_w_m)
    term3 = 2*ln_w_m
    total = term1 + term2 + term3
    if total <= eps*theta:
        return 1
    else:
        return 0


m = 0
if args.exact == 0:

    df = pd.read_csv(args.db, nrows=10)
    d = df.columns.shape[0]
    if args.v > 0:
        print("number of features is", d)

    m_lb = 3*math.log(2./delta)/theta
    m_ub = m_lb

    while check_current_sample_size(m_ub) == 0:
        m_ub = m_ub*2

    while m_ub-m_lb > 1.:
        m = (m_ub+m_lb)/2.
        test_check = check_current_sample_size(m)
        if test_check == 0:
            m_lb = m
        else:
            m_ub = m

    m = math.ceil(m_ub)
    if args.v > 0:
        print("sample size is", m)


if args.exact == 0:
    cmd = f"python3 sample_and_run.py -db {args.db} -k {args.k} -z {args.z} -minf {args.minf} -r {args.r} -m {m} -op {args.op}"
else:
    cmd = f"python3 sample_and_run.py -db {args.db} -k {args.k} -z {args.z} -minf {args.minf} -r {args.r} -s {args.exact} -op {args.op}"
cmd = f"{cmd} -v {args.v} -f {args.f}"

if args.seed is not None:
    cmd = f"{cmd} -seed {args.seed}"

if args.v > 0:
    print(cmd)
ret_val = os.system(cmd)

if ret_val > 0:
    print("ERROR with sample_and_run.py!!!")
    print(cmd)
    exit(1)


base_name = os.path.basename(args.op) if args.op else ""
out_file_path = os.path.join(corelsOut_dir, f"{base_name}out.txt")
fin = open(out_file_path, "r")
running_time = 0.
min_objective = 0.
optimal_rule = ""
for line in fin:
    parse_term = "final total time: "
    if parse_term in line:
        line_parsed = line.replace("\n", "")
        line_parsed = line_parsed.replace(parse_term, "")
        running_time = float(line_parsed)
        if args.v > 0:
            print("running time:", running_time)
    parse_term = "final min_objective: "
    if parse_term in line:
        line_parsed = line.replace("\n", "")
        line_parsed = line_parsed.replace(parse_term, "")
        min_objective = float(line_parsed)
        if args.v > 0:
            print("min_objective:", min_objective)
    parse_term = "OPTIMAL RULE LIST"
    if parse_term in line:
        line_parsed = line.replace("\n", "")
        opt_rule_str = ""
        while len(line) > 0:
            line = fin.readline()
            line = line.replace("\n", "")
            if len(line) > 0:
                opt_rule_str = opt_rule_str+line+", "
        last_comma_pos = opt_rule_str.rfind(',')
        opt_rule_str = opt_rule_str[:last_comma_pos]
        optimal_rule = opt_rule_str
        if args.v > 0:
            print("optimal rule:", opt_rule_str)


res_file_path = args.ores
if not os.path.isfile(res_file_path):
    fout = open(res_file_path, "w")
    fout.write(
        "dataset;k;z;exact;theta;epsilon;delta;alpha;m;running_time;min_loss;opt_rule\n")
    fout.close()
fout = open(res_file_path, "a")
res = f"{args.db};{k};{z};{args.exact};{theta};{eps};{delta};{args.r};{m};{running_time};{min_objective};{opt_rule_str}"
fout.write(res+"\n")
fout.close()
