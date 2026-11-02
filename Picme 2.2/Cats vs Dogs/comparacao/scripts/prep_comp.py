# Roda as células 2 a 16 (100 x 94 e divisão) e monta a versão 224 x 224 com as mesmas imagens e a mesma divisão.
import sys, site, time
sys.path.append(site.getusersitepackages())
import nbformat, numpy as np, torch
from PIL import Image
p = "/home/al.dulce.costa/Machine-Learning/Picme 2.2/Cats vs Dogs/cats_and_dogs.ipynb"
nb = nbformat.read(p, as_version=4)
g = {}
for n in range(2, 17):
    t = time.time(); exec(nb.cells[n-1].source, g); print(f"[célula {n}: {time.time()-t:.0f}s]", flush=True)
assert (g["LARGURA"], g["ALTURA"]) == (100, 94)
comum = {"y": g["y"], "treino": torch.as_tensor(g["treino"]), "validacao": torch.as_tensor(g["validacao"]), "teste": torch.as_tensor(g["teste"]), "grupo": torch.as_tensor(g["grupo"])}
torch.save(comum, "comum.pt"); torch.save(g["X"], "X_94x100.pt"); del g["X"], g["dados"]
print("100x94 salvo", flush=True)
g["ALTURA"], g["LARGURA"] = 224, 224
total = len(comum["y"]); X = np.empty((total, 3, 224, 224), np.float32); k = 0; t = time.time(); ys = []
for i, d in enumerate(g["imagens"]):
    with Image.open(d["arquivo"]) as im:
        img = np.asarray(im.convert("RGB"), dtype=np.float32)
    base = g["redimensionar"](img); versoes = [base]
    if i in g["vao_girar"]: versoes.append(g["redimensionar"](g["girar_horario"](img)))
    elif i in g["vao_ruido"]: versoes.append(g["adicionar_ruido"](base))
    for v in versoes:
        X[k] = v.transpose(2, 0, 1) / 255; ys.append(g["classes"].index(d["classe"])); k += 1
assert k == total and ys == comum["y"].tolist()
torch.save(torch.from_numpy(X), "X_224x224.pt"); print(f"224x224 salvo: {X.shape} em {time.time()-t:.0f}s, min {X.min()} max {X.max()}", flush=True)
