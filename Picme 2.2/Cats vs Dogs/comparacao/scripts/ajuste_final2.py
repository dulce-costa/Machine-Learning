import os, base64, warnings, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient
warnings.simplefilter("ignore")
PASTA = "/home/al.dulce.costa/Machine-Learning/Picme 2.2/Cats vs Dogs"
DEST = PASTA + "/comparacao_modelos.ipynb"
TROCAS = [
    ('ax_a.legend(frameon=False, loc="upper left", fontsize=9.5)', 'ax_a.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.1), fontsize=9.5)'),
    ('max(np.mean(v), max(v)) + 0.0012', 'max(v) + 0.002'),
]
nb = nbformat.read(DEST, as_version=4)
achar = lambda t: next(c for c in nb.cells if c.cell_type == "code" and t in c.source)
for a, b in TROCAS:
    c = achar(a); c.source = c.source.replace(a, b)
figs = [achar("X_MIN, X_MAX"), achar("fig, (ax_a, ax_p)")]
parcial = [nb.cells[1], achar('PASTA_RES = "comparacao"'), achar("def nome_arquivo"), achar("resultados = {}")] + figs
tmp = new_notebook(cells=[new_code_cell(c.source) for c in parcial]); tmp.metadata = nb.metadata
NotebookClient(tmp, timeout=600, kernel_name="ambiente_aluno", resources={"metadata": {"path": PASTA}}).execute()
for k, (c, t) in enumerate(zip(figs, tmp.cells[-2:])):
    n = c.execution_count; c.outputs = t.outputs
    for o in c.outputs:
        if "execution_count" in o: o.execution_count = n
        if "data" in o and "image/png" in o.data:
            open(f"final_{k}.png", "wb").write(base64.b64decode(o.data["image/png"]))
nbformat.validate(nb)
nbformat.write(nb, DEST)
print("gravado", len(nb.cells), "células")
