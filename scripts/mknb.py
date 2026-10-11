"""Construtor de notebooks: executa células em sequência e grava texto e gráficos como saídas do .ipynb.

Usado pelos scripts notebooks/construir_*.py para que cada notebook publicado seja reproduzível.
"""
import json, io, os, contextlib, base64, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build(cells, out, cwd):
    ns = {}
    nb = {"cells": [], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    old = os.getcwd(); os.chdir(cwd); n = 0
    import sys as _s; _s.path.insert(0, os.path.abspath(cwd))
    try:
        for kind, src in cells:
            src = src.strip("\n")
            if kind == "md":
                nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}); continue
            n += 1; buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r = exec(compile(src, f"<cell{n}>", "exec"), ns)
            outs = []
            if buf.getvalue():
                outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
            for num in plt.get_fignums():
                fig = plt.figure(num); b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
                outs.append({"output_type": "display_data", "metadata": {},
                             "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
            plt.close("all")
            nb["cells"].append({"cell_type": "code", "execution_count": n, "metadata": {}, "source": src.splitlines(True), "outputs": outs})
    finally:
        os.chdir(old)
    json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
