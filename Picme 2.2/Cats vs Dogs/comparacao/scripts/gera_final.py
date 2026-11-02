src = open("nb_comp.py", encoding="utf8").read()
corpo = src[:src.index("NB = new_notebook(cells=cels)")]
def troca(a, b):
    global corpo
    assert corpo.count(a) == 1, a
    corpo = corpo.replace(a, b)
troca("X_MIN, X_MAX = 0.89, 0.93", "X_MIN, X_MAX = 0.80, 0.95")
troca('ax.text(media_ref + 0.0005, len(ordem_cfg) - 0.42, "faixa de até +1 ponto\\nsobre a referência", fontsize=9, color=TINTA_2, va="top")',
      'ax.text(media_ref + 0.012, 2, "faixa de até +1 ponto\\nsobre a referência", fontsize=9, color=TINTA_2, va="center")')
troca('''    if not v:
        ax.text(X_MIN + 0.0005, i, "treinando, ainda sem resultado", va="center", fontsize=10, color=TINTA_2, style="italic")
    elif max(v) < X_MIN:''', "    if max(v) < X_MIN:")
troca('np.arange(X_MIN, X_MAX + 0.001, 0.01)', 'np.arange(X_MIN, X_MAX + 0.001, 0.02)')
troca('''faltam = [c for c in CONFIGS if not melhores(c)]
''', '')
troca('''
print("Ainda sem resultado:", [CONFIGS[c]["nome"] for c in faltam])''', '')
corpo += r'''
C_VAL = code(r"""assert len(vencedoras) == 1   # com duas ou mais vencedoras, a regra pediria treinar a combinação
config_final = vencedoras[0]
seed_final = max(SEEDS, key=lambda s: resultados[config_final, s]["melhor_acc"])
c = CONFIGS[config_final]

final = criar_rede(c["altura"], c["largura"], c["padding"], c["convs"]).to(dispositivo)
final.load_state_dict(torch.load(nome_arquivo(config_final, seed_final) + ".pt", weights_only=True))
final.eval()

def prever(indices):
    previsoes = []
    with torch.no_grad():
        for i in range(0, len(indices), 256):
            previsoes.append(final(X[indices[i:i + 256]].to(dispositivo)).argmax(1).cpu())
    return torch.cat(previsoes)

# Confere se os pesos e os dados deste notebook reproduzem o resultado salvo, antes de olhar o teste.
acc_val = (prever(validacao) == y[validacao]).float().mean().item()
salvo = resultados[config_final, seed_final]["melhor_acc"]
print(f"Vencedor: {c['nome']}, seed {seed_final}")
print(f"Validação recalculada: {acc_val:.4f} | salva no treino: {salvo:.4f}")
assert abs(acc_val - salvo) < 0.0005""")
C_TESTE = code(r"""previsto_final = prever(teste)
real = y[teste]
matriz_final = np.zeros((2, 2), dtype=int)
for r_, p_ in zip(real.tolist(), previsto_final.tolist()):
    matriz_final[r_, p_] += 1

# Modelo fechado: números já medidos em cats_and_dogs.ipynb, células 20 e 22.
matriz_fechado = np.array([[1966, 232], [170, 2100]])

margem = lambda m: 1.96 * np.sqrt((m.trace() / m.sum()) * (1 - m.trace() / m.sum()) / m.sum())
print(f"{'Modelo':32s} {'acertos':>12s} {'acurácia':>9s} {'margem 95%':>11s}")
for nome, m in [("Fechado (cats_and_dogs.ipynb)", matriz_fechado), (c["nome"], matriz_final)]:
    print(f"{nome:32s} {m.trace():5d} de {m.sum():4d} {m.trace() / m.sum():9.4f} {100 * margem(m):8.1f} p")""")
C_MATRIZ = code(r"""from matplotlib.colors import LinearSegmentedColormap
rotulos = ["gato", "cachorro"]
azuis = LinearSegmentedColormap.from_list("azuis", ["#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
fig, eixos = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, m, titulo in zip(eixos, [matriz_fechado, matriz_final], ["Modelo fechado", c["nome"]]):
    ax.imshow(m, cmap=azuis, vmin=0, vmax=max(matriz_fechado.max(), matriz_final.max()))
    for i in range(2):
        for j in range(2):
            cor = "white" if m[i, j] > m.max() / 2 else TINTA
            ax.text(j, i - 0.07, f"{m[i, j]}", ha="center", va="center", color=cor, fontsize=22, fontweight="bold")
            ax.text(j, i + 0.13, f"{m[i, j] / m[i].sum():.1%} dos {rotulos[i]}s".replace(".", ","), ha="center", va="center", color=cor, fontsize=10)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=rotulos, yticklabels=rotulos, xlabel="Classe prevista", ylabel="Classe real", title=titulo)
    ax.grid(False)
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
fig.suptitle("Matrizes de confusão no teste", x=0.01, ha="left", fontweight="bold", fontsize=14)
fig.tight_layout()
plt.show()""")
cels += [md("""## 6. Teste do vencedor
Só a configuração vencedora é medida no teste, na seed com a melhor validação. As outras configurações não são medidas. O resultado do teste não muda a escolha, que já foi feita pela validação."""), C_VAL, C_TESTE, C_MATRIZ]
cels[1].source = "import glob\n" + cels[1].source
NB = new_notebook(cells=cels)
NB.metadata = base.metadata
cliente = NotebookClient(NB, timeout=None, kernel_name="ambiente_aluno", resources={"metadata": {"path": PASTA}})
try:
    cliente.execute()
finally:
    nbformat.write(NB, "final_executado.ipynb")
nbformat.validate(NB)
nbformat.write(NB, DEST)
k = 0
for c in NB.cells:
    if c.cell_type == "code":
        for o in c.outputs:
            if "data" in o and "image/png" in o.data:
                open(f"final_{k}.png", "wb").write(base64.b64decode(o.data["image/png"])); k += 1
for c in NB.cells[-5:]:
    if c.cell_type == "code":
        for o in c.outputs:
            if o.output_type == "stream": print(o.text)
print("gravado", len(NB.cells), "células")
'''
open("nb_final.py", "w", encoding="utf8").write(corpo)
