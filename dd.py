#!/usr/bin/env python3
# find_unused_config_vars_py36.py
import argparse
import ast
import pathlib
from collections import Counter

# ← plus d'annotations génériques
def load_var_names(path):
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    }

class VarCounter(ast.NodeVisitor):
    def __init__(self, targets):
        self.targets = targets
        self.counts = Counter()

    def visit_Name(self, node):
        if node.id in self.targets:
            self.counts[node.id] += 1
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in self.targets:
            self.counts[node.attr] += 1
        self.generic_visit(node)

def scan_py_file(path, targets):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError):
        return Counter()
    vc = VarCounter(targets)
    vc.visit(tree)
    return vc.counts

def walk_python_files(root):
    return [
        p for p in root.rglob("*.py")
        if p.name != "config.py" and p.is_file()
    ]

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-v", "--vars", default="config.vars.txt",
                    help="fichier contenant les variables")
    ap.add_argument("-r", "--root", default=".",
                    help="répertoire racine à analyser")
    args = ap.parse_args()

    var_file = pathlib.Path(args.vars).expanduser().resolve()
    root_dir = pathlib.Path(args.root).expanduser().resolve()

    if not var_file.exists():
        raise SystemExit("Fichier introuvable : %s" % var_file)
    if not root_dir.exists():
        raise SystemExit("Dossier introuvable : %s" % root_dir)

    targets = load_var_names(var_file)
    if not targets:
        raise SystemExit("Aucune variable à tester.")

    total = Counter()
    for py in walk_python_files(root_dir):
        total.update(scan_py_file(py, targets))

    unused = sorted([v for v in targets if total[v] == 0])

    print("Variables jamais vues (%d/%d) :" % (len(unused), len(targets)))
    for v in unused:
        print("  ", v)

    print("\nVariables rencontrées au moins une fois :")
    for v, n in total.most_common():
        print("  %-35s %d" % (v, n))

if __name__ == "__main__":
    main()
