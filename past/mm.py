import ast
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ls", type=str)
args = parser.parse_args()
ls = args.ls

print(ls)
print(type(ls))

aa = ast.literal_eval(ls)
print(aa)
print(type(aa))
