import sys, os, shutil, glob, base64, warnings, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient
warnings.simplefilter("ignore")
PASTA = "/home/al.dulce.costa/Machine-Learning/Picme 2.2/Cats vs Dogs"
DEST = PASTA + "/comparacao_modelos.ipynb"; RES = PASTA + "/comparacao"
os.makedirs(RES + "/batch", exist_ok=True)
for f in glob.glob("res/*_b32_lr0.01_s*.json"):   # só rodadas terminadas, com seus pesos
    shutil.copy2(f, RES); shutil.copy2(f[:-5] + ".pt", RES)
for f in glob.glob("res/referencia_b*.json"):
    if "_b32_" not in f: shutil.copy2(f, RES + "/batch")

base = nbformat.read(PASTA + "/cats_and_dogs.ipynb", as_version=4)
md = new_markdown_cell; code = new_code_cell
cels = [md("""# Comparação de modelos: Cats and Dogs

Treina variações do modelo fechado (`cats_and_dogs.ipynb`) para ver o efeito de cada mudança. Cada variação muda **uma coisa só** em relação à referência e roda com 3 seeds, sempre com batch 32, learning rate 0,01 e 100 épocas.

**Regras:** a comparação usa só a validação. O teste é medido uma única vez, só no vencedor. Este notebook não altera nada do `cats_and_dogs.ipynb` nem o `modelo_cats_and_dogs.pt`.

Os resultados ficam na pasta `comparacao/`. Com `TREINAR_O_QUE_FALTA = False`, nada é treinado de novo: as células só carregam o que já foi salvo.

## 1. Dados
Mesmas células do `cats_and_dogs.ipynb`, na mesma ordem e com a mesma seed, para a divisão ser idêntica.""")]
cels += [code(base.cells[n - 1].source) for n in range(2, 17)]
IDX_IMPORT = 1
cels.append(md("""## 2. Configurações
A imagem de 224 × 224 usa as mesmas fotos e a mesma divisão. Ela só é montada quando precisa, porque ocupa cerca de 17 GB de memória."""))
C_CFG = code(r'''import json
import matplotlib.pyplot as plt

PASTA_RES = "comparacao"
TREINAR_O_QUE_FALTA = False   # True treina as configurações que ainda não têm resultado salvo
SEEDS = [18, 19, 20]
EPOCAS, BATCH, LR = 100, 32, 0.01

# altura, largura, padding, convoluções por bloco, espelhar a imagem ao acaso no treino
CONFIGS = {
    "referencia":      dict(nome="Referência (modelo atual)",    altura=94,  largura=100, padding=1, convs=1, espelhar=False),
    "padding2":        dict(nome="Padding 2 (saída 2×3)",        altura=94,  largura=100, padding=2, convs=1, espelhar=False),
    "camadas10":       dict(nome="10 camadas convolucionais",    altura=94,  largura=100, padding=2, convs=2, espelhar=False),
    "imagem224":       dict(nome="Imagem 224×224 (saída 5×5)",   altura=224, largura=224, padding=1, convs=1, espelhar=False),
    "referencia_flip": dict(nome="Espelhamento a cada época",    altura=94,  largura=100, padding=1, convs=1, espelhar=True),
}

def criar_rede(altura, largura, padding, convs):
    blocos = []
    canais = 3
    for _ in range(5):
        for _ in range(convs):
            blocos += [nn.Conv2d(canais, 32, kernel_size=5, stride=1, padding=padding), nn.ReLU()]
            canais = 32
            altura += 2 * padding - 4
            largura += 2 * padding - 4
        blocos += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.2)]
        altura //= 2
        largura //= 2
    return nn.Sequential(*blocos, nn.Flatten(),
                         nn.Linear(32 * altura * largura, 1024), nn.ReLU(),
                         nn.Linear(1024, 32), nn.ReLU(),
                         nn.Linear(32, 2))''')
C_X = code(r'''dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
_X224 = None

def montar_224():
    # Mesmas linhas de X (célula de montagem), agora em 224 x 224. Só o sorteio do ruído muda.
    global _X224
    if _X224 is None:
        _X224 = torch.empty((len(y), 3, 224, 224))
        global ALTURA, LARGURA
        guardado = ALTURA, LARGURA
        ALTURA, LARGURA = 224, 224
        k = 0
        for i, d in enumerate(imagens):
            with Image.open(d["arquivo"]) as im:
                img = np.asarray(im.convert("RGB"), dtype=np.float32)
            base = redimensionar(img)
            versoes = [base]
            if i in vao_girar:
                versoes.append(redimensionar(girar_horario(img)))
            elif i in vao_ruido:
                versoes.append(adicionar_ruido(base))
            for v in versoes:
                _X224[k] = torch.from_numpy(v).permute(2, 0, 1) / 255
                k += 1
        ALTURA, LARGURA = guardado
        assert k == len(y)
    return _X224

def dados_de(config):
    return montar_224() if CONFIGS[config]["altura"] == 224 else X''')
C_TR = code(r'''def nome_arquivo(config, seed):
    return f"{PASTA_RES}/{config}_b{BATCH}_lr{LR:g}_s{seed}"

def treinar(config, seed):
    # Usa só treino e validação. O teste não entra aqui.
    c = CONFIGS[config]
    Xc = dados_de(config)
    torch.manual_seed(seed)
    m = criar_rede(c["altura"], c["largura"], c["padding"], c["convs"]).to(dispositivo)
    otim = torch.optim.SGD(m.parameters(), lr=LR)
    tr, va = torch.as_tensor(treino), torch.as_tensor(validacao)
    hist = dict(perda_treino=[], acc_treino=[], perda_val=[], acc_val=[])
    melhor_acc, melhor_ep = 0, 0
    for epoca in range(1, EPOCAS + 1):
        m.train()
        ordem_ep = tr[torch.randperm(len(tr))]
        soma, acertos = 0.0, 0
        for i in range(0, len(ordem_ep), BATCH):
            k = ordem_ep[i:i + BATCH]
            xb, yb = Xc[k].to(dispositivo), y[k].to(dispositivo)
            if c["espelhar"]:
                sorteio = torch.rand(len(k), device=dispositivo) < 0.5
                xb[sorteio] = xb[sorteio].flip(-1)
            z = m(xb)
            perda = F.cross_entropy(z, yb)
            otim.zero_grad()
            perda.backward()
            otim.step()
            soma += perda.item() * len(k)
            acertos += (z.argmax(1) == yb).sum().item()
        m.eval()
        soma_val, acertos_val = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va), 256):
                k = va[i:i + 256]
                xb, yb = Xc[k].to(dispositivo), y[k].to(dispositivo)
                z = m(xb)
                soma_val += F.cross_entropy(z, yb, reduction="sum").item()
                acertos_val += (z.argmax(1) == yb).sum().item()
        acc = acertos_val / len(va)
        hist["perda_treino"].append(soma / len(tr))
        hist["acc_treino"].append(acertos / len(tr))
        hist["perda_val"].append(soma_val / len(va))
        hist["acc_val"].append(acc)
        if acc > melhor_acc:
            melhor_acc, melhor_ep = acc, epoca
            torch.save({k: v.cpu() for k, v in m.state_dict().items()}, nome_arquivo(config, seed) + ".pt")
    resultado = dict(cfg=config, seed=seed, batch=BATCH, lr=LR, melhor_acc=melhor_acc, melhor_epoca=melhor_ep, hist=hist)
    json.dump(resultado, open(nome_arquivo(config, seed) + ".json", "w"))
    return resultado''')
C_RES = code(r'''resultados = {}
for config in CONFIGS:
    for seed in SEEDS:
        arq = nome_arquivo(config, seed) + ".json"
        if os.path.exists(arq):
            resultados[config, seed] = json.load(open(arq))
        elif TREINAR_O_QUE_FALTA:
            resultados[config, seed] = treinar(config, seed)

def melhores(config):
    return [resultados[config, s]["melhor_acc"] for s in SEEDS if (config, s) in resultados]

media_ref = np.mean(melhores("referencia"))
print(f"{'Configuração':32s} {'seed 18':>8s} {'seed 19':>8s} {'seed 20':>8s} {'média':>7s} {'vs ref.':>8s}")
for config, c in CONFIGS.items():
    v = melhores(config)
    if not v:
        print(f"{c['nome']:32s}   ainda sem resultado")
        continue
    print(f"{c['nome']:32s} " + " ".join(f"{a:8.3f}" for a in v) + f" {np.mean(v):7.3f} {100 * (np.mean(v) - media_ref):+7.1f} p")''')
C_FIG1 = code(r'''AZUL, CINZA, TINTA, TINTA_2, GRADE, FUNDO = "#2a78d6", "#8a8985", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
CORES = dict(referencia="#2a78d6", padding2="#eb6834", camadas10="#1baf7a", imagem224="#eda100", referencia_flip="#e87ba4")
plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "text.color": TINTA, "axes.labelcolor": TINTA_2,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.edgecolor": GRADE, "axes.grid": True,
    "grid.color": GRADE, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
virgula = lambda v, _=None: f"{100 * v:.0f}%".replace(".", ",")

ordem_cfg = list(CONFIGS)[::-1]
X_MIN, X_MAX = 0.89, 0.93
fig, ax = plt.subplots(figsize=(10.5, 4.2))
ax.axvspan(media_ref, media_ref + 0.01, color=GRADE, alpha=0.7, lw=0)
ax.axvline(media_ref, color=TINTA_2, lw=1)
ax.text(media_ref + 0.0005, len(ordem_cfg) - 0.42, "faixa de até +1 ponto\nsobre a referência", fontsize=9, color=TINTA_2, va="top")
for i, config in enumerate(ordem_cfg):
    v = melhores(config)
    if not v:
        ax.text(X_MIN + 0.0005, i, "treinando, ainda sem resultado", va="center", fontsize=10, color=TINTA_2, style="italic")
    elif max(v) < X_MIN:
        ax.text(X_MIN + 0.0005, i, f"não aprendeu: {100 * np.mean(v):.1f}% (fora da escala)".replace(".", ","), va="center", fontsize=10, color=TINTA_2, style="italic")
    else:
        ax.scatter(v, [i] * len(v), s=80, color=AZUL, edgecolor=FUNDO, linewidth=2, zorder=3, label="Cada seed" if config == "referencia" else None)
        ax.plot([np.mean(v)] * 2, [i - 0.28, i + 0.28], color=TINTA, lw=2.5, zorder=4, label="Média" if config == "referencia" else None)
        ax.text(max(np.mean(v), max(v)) + 0.0012, i, f"{100 * np.mean(v):.1f}%".replace(".", ","), va="center", fontsize=10.5, fontweight="bold")
ax.set_yticks(range(len(ordem_cfg)), [CONFIGS[c]["nome"] for c in ordem_cfg])
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(-0.6, len(ordem_cfg) - 0.4)
ax.set_xticks(np.arange(X_MIN, X_MAX + 0.001, 0.01))
ax.xaxis.set_major_formatter(virgula)
ax.grid(axis="y", visible=False)
ax.tick_params(length=0)
ax.set_title("Melhor acurácia na validação, por configuração")
ax.set_xlabel("Acurácia na validação (o eixo não começa em zero)")
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2, fontsize=9.5)
fig.tight_layout()
plt.show()''')
C_FIG2 = code(r'''fig, (ax_a, ax_p) = plt.subplots(1, 2, figsize=(13, 4.8))
epocas = np.arange(1, EPOCAS + 1)
for config, c in CONFIGS.items():
    runs = [resultados[config, s]["hist"] for s in SEEDS if (config, s) in resultados]
    if not runs:
        continue
    ax_a.plot(epocas, np.mean([r["acc_val"] for r in runs], axis=0), color=CORES[config], lw=2, label=c["nome"])
    ax_p.plot(epocas, np.mean([r["perda_val"] for r in runs], axis=0), color=CORES[config], lw=2, label=c["nome"])
ax_a.axhline(0.5, color=TINTA_2, lw=1, ls=":")
ax_a.set(title="Acurácia na validação (média das seeds)", xlabel="Época", ylabel="Acurácia", xlim=(1, EPOCAS), ylim=(0.45, 1))
ax_a.yaxis.set_major_formatter(virgula)
ax_a.legend(frameon=False, loc="lower right", fontsize=9.5)
ax_p.set(title="Perda na validação (média das seeds)", xlabel="Época", ylabel="Perda", xlim=(1, EPOCAS), ylim=(0, None))
ax_p.legend(frameon=False, loc="lower left", fontsize=9.5)
fig.tight_layout()
plt.show()''')
C_BATCH = code(r'''linhas = [json.load(open(f)) for f in glob.glob(f"{PASTA_RES}/batch/*.json")]
print(f"{'Batch':>6s} {'Learning rate':>14s} {'Melhor val.':>12s} {'Val. final':>11s}")
print(f"{32:6d} {0.01:14g} {melhores('referencia')[0]:12.3f} {resultados['referencia', 18]['hist']['acc_val'][-1]:11.3f}   (referência, seed 18)")
for d in sorted(linhas, key=lambda d: (d["batch"], d["lr"])):
    print(f"{d['batch']:6d} {d['lr']:14g} {d['melhor_acc']:12.3f} {d['hist']['acc_val'][-1]:11.3f}")''')
C_REGRA = code(r'''vencedoras = [c for c in CONFIGS if c != "referencia" and melhores(c) and np.mean(melhores(c)) > media_ref + 0.01]
faltam = [c for c in CONFIGS if not melhores(c)]
print("Superaram a referência por mais de 1 ponto:", [CONFIGS[c]["nome"] for c in vencedoras])
print("Ainda sem resultado:", [CONFIGS[c]["nome"] for c in faltam])''')
cels += [C_CFG, C_X, C_TR, md("## 3. Resultados na validação\nCom `TREINAR_O_QUE_FALTA = False`, esta célula só carrega o que está salvo em `comparacao/`."), C_RES, C_FIG1, C_FIG2,
         md("""## 4. Efeito do tamanho do batch
Na configuração da referência, uma seed por linha. Com batch grande, a rede dá poucos passos por época e não sai de 50%, e aumentar o learning rate na mesma proporção só funcionou com batch 256. O batch maior também quase não acelerou o treino, por isso a comparação usa batch 32."""), C_BATCH,
         md("## 5. O que entra na combinação\nRegra combinada: entram só as mudanças que superam a referência por mais de 1 ponto na média das seeds."), C_REGRA]
cels[1].source = "import glob\n" + cels[1].source
NB = new_notebook(cells=cels)
NB.metadata = base.metadata

# Execução parcial: só o que não depende de rodar a preparação dos dados.
parcial = [cels[1], C_CFG, C_TR, C_RES, C_FIG1, C_FIG2, C_BATCH, C_REGRA]
TMP = new_notebook(cells=parcial); TMP.metadata = base.metadata
NotebookClient(TMP, timeout=600, kernel_name="ambiente_aluno", resources={"metadata": {"path": PASTA}}).execute()
for k, c in enumerate(parcial):
    for o in c.outputs:
        if o.output_type == "stream" and c is not cels[1]: print(o.text)
        if "data" in o and "image/png" in o.data:
            open(f"comp_{k}.png", "wb").write(base64.b64decode(o.data["image/png"]))
cels[1].outputs = []; cels[1].execution_count = None
for c in NB.cells:
    if c.cell_type == "code" and c.get("execution_count"): c.execution_count = None
for c in NB.cells:
    if c.cell_type == "code":
        for o in c.get("outputs", []):
            if "execution_count" in o: o.execution_count = None
nbformat.validate(NB)
nbformat.write(NB, DEST)
print("gravado", len(NB.cells), "células")
