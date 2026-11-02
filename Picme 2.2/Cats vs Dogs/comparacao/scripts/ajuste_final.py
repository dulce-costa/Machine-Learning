import os, base64, warnings, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient
warnings.simplefilter("ignore")
PASTA = "/home/al.dulce.costa/Machine-Learning/Picme 2.2/Cats vs Dogs"
DEST = PASTA + "/comparacao_modelos.ipynb"
TROCAS = [
    ("X_MIN, X_MAX = 0.80, 0.95", "X_MIN, X_MAX = 0.80, 0.96"),
    ('f"{100 * np.mean(v):.1f}%".replace(".", ","), va="center", fontsize=10.5, fontweight="bold")', 'f"média {100 * np.mean(v):.1f}%".replace(".", ","), va="center", fontsize=10.5, fontweight="bold")'),
    ('ax_a.legend(frameon=False, loc="lower right", fontsize=9.5)', 'ax_a.legend(frameon=False, loc="upper left", fontsize=9.5)'),
    ('ax_p.legend(frameon=False, loc="lower left", fontsize=9.5)\n', ''),
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
nb.cells.append(new_markdown_cell("""## 7. Conclusão

| Modelo | Validação | Teste |
|---|---|---|
| Fechado (`modelo_cats_and_dogs.pt`) | 90,5% | 91,0% (4066 de 4468) |
| **Espelhamento a cada época, seed 18** | **92,1%** | **91,6% (4094 de 4468)** |

- **Vencedor:** espelhamento a cada época, a única mudança que superou a referência por mais de 1 ponto na média da validação (+1,2 ponto). Pela regra combinada, ele passa a ser o modelo oficial. Os pesos estão em `comparacao/referencia_flip_b32_lr0.01_s18.pt`.
- **Combinação:** não foi treinada, porque só uma mudança venceu.
- **Teste:** a diferença de 0,6 ponto é menor que a margem de cerca de 0,8 ponto de cada medida, então o teste sozinho não comprova a melhora. O erro ficou mais equilibrado: 8,4% nas duas classes, contra 10,6% nos gatos e 7,5% nos cachorros no modelo fechado.
- **Padding 2:** não teve efeito (-0,1 ponto).
- **Imagem 224 × 224:** perdeu na média (-2,0 pontos) e foi muito instável, de 81,4% a 93,3% conforme a seed. Na seed 18, a perda de validação piorou a partir da época 46 enquanto a de treino continuava caindo, sinal de overfitting. O teste não foi medido nessa configuração.
- **10 camadas:** não aprendeu nas três seeds, com perda parada em 0,693. Isso vale para esta configuração, com SGD e learning rate 0,01, e não prova que mais camadas pioram o modelo.
- **Saída 1 × 1:** os resultados não indicam que ela fosse o principal problema, porque o padding 2 não mudou nada e a imagem maior não melhorou de forma consistente."""))
nbformat.validate(nb)
nbformat.write(nb, DEST)
print("gravado", len(nb.cells), "células")
