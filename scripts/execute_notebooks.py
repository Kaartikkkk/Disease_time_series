#!/usr/bin/env python3
"""Execute Jupyter notebooks and embed real stdout, dataframe tables, and matplotlib figures."""

import ast
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import traceback

# Ensure current directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def run_notebook(nb_path: Path, output_path: Path):
    print(f"\n==========================================")
    print(f"Executing: {nb_path.name}")
    print(f"==========================================")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Isolated execution namespace for this notebook
    ns = {
        "__name__": "__main__",
        "__file__": str(nb_path.resolve()),
    }

    execution_count = 1

    for cell_idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])
        print(f"  [Cell {cell_idx+1}] Running...", end="", flush=True)

        cell["execution_count"] = execution_count
        cell["outputs"] = []

        stdout_buf = io.StringIO()
        plt.close("all")

        try:
            saved_figures = []
            orig_show = plt.show

            def custom_show(*args, **kwargs):
                fig = plt.gcf()
                if fig.get_axes():
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=100)
                    img_buf.seek(0)
                    b64_str = base64.b64encode(img_buf.read()).decode("utf-8")
                    saved_figures.append(b64_str)
                orig_show(*args, **kwargs)

            plt.show = custom_show

            # Execute cell source with ast to support expression outputs
            parsed = ast.parse(source)
            last_expr = None
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                last_expr = parsed.body.pop()

            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
                if parsed.body:
                    exec(compile(parsed, filename="<string>", mode="exec"), ns)
                eval_res = None
                if last_expr is not None:
                    eval_res = eval(compile(ast.Expression(last_expr.value), filename="<string>", mode="eval"), ns)

            plt.show = orig_show

            # If there is un-shown active figure, grab it too
            fig = plt.gcf()
            if fig.get_axes() and not saved_figures:
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=100)
                img_buf.seek(0)
                b64_str = base64.b64encode(img_buf.read()).decode("utf-8")
                saved_figures.append(b64_str)
            plt.close("all")

            # Check stdout
            stdout_text = stdout_buf.getvalue()
            if stdout_text:
                cell["outputs"].append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": stdout_text.splitlines(keepends=True)
                })

            # Check eval result (dataframe or object)
            if eval_res is not None:
                plain_text = repr(eval_res)
                data_dict = {"text/plain": [plain_text + "\n"]}
                if isinstance(eval_res, pd.DataFrame):
                    data_dict["text/html"] = [eval_res._repr_html_()]
                cell["outputs"].append({
                    "output_type": "execute_result",
                    "execution_count": execution_count,
                    "data": data_dict,
                    "metadata": {}
                })

            # Check figures
            for b64 in saved_figures:
                cell["outputs"].append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": b64,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {}
                })

            print(" [OK]")
            execution_count += 1

        except Exception as e:
            plt.show = orig_show
            plt.close("all")
            print(f" [ERROR: {e}]")
            err_msg = traceback.format_exc()
            cell["outputs"].append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": err_msg.splitlines()
            })
            raise

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully saved pre-run notebook to: {output_path}")

if __name__ == "__main__":
    notebooks = [
        Path("notebooks/01_data_acquisition_and_eda.ipynb"),
        Path("notebooks/02_classical_econometrics_and_multivariate.ipynb"),
        Path("notebooks/03_machine_learning_deep_learning_and_autots.ipynb"),
    ]
    for nb in notebooks:
        run_notebook(nb, nb)
