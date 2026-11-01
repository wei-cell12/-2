"""CPU-only recomputation; requires the original symexp checkout and server-side arrays."""
import argparse
from symexp.fp.reading_groups import run_groups,outcomes_clock,CQ_GROUPS
p=argparse.ArgumentParser();p.add_argument('--cq',required=True);p.add_argument('--cache',required=True);p.add_argument('--judgments',required=True);p.add_argument('--out-md',required=True);p.add_argument('--out-json',required=True);a=p.parse_args()
bel,neu=outcomes_clock(a.judgments)
run_groups(a.cq,a.cache,bel,neu,CQ_GROUPS,'clockqa',out_md=a.out_md,out_json=a.out_json)
