# Treina uma configuração e registra o histórico. Só usa treino e validação; nunca o teste.
import sys, json, time, torch, torch.nn.functional as F
from comp_lib import rede, CFG
cfg, seed, B, lr, flip, E = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5] == "flip", int(sys.argv[6])
d = "cuda"; H, W, p, c = CFG[cfg]; nome = f"{cfg}{'_flip' if flip else ''}_b{B}_lr{lr:g}_s{seed}"
K = torch.load("comum.pt", weights_only=True); Y = K["y"]; tr, va = K["treino"], K["validacao"]
X = torch.load("X_224x224.pt" if H == 224 else "X_94x100.pt", weights_only=True, mmap=H == 224)
if H != 224 and B <= 512: X, Y, tr, va = X.to(d), Y.to(d), tr.to(d), va.to(d)
torch.manual_seed(seed)
m, _ = rede(H, W, p, c); m = m.to(d); o = torch.optim.SGD(m.parameters(), lr=lr)
H_ = dict(perda_treino=[], acc_treino=[], perda_val=[], acc_val=[]); melhor = (0, 0); t0 = time.time()
for ep in range(1, E + 1):
    m.train(); q = tr[torch.randperm(len(tr), device=tr.device)]; sp = ac = 0
    for i in range(0, len(q), B):
        k = q[i:i+B]; xb, yb = X[k].to(d), Y[k].to(d)
        if flip:
            f = torch.rand(len(k), device=d) < .5; xb[f] = xb[f].flip(-1)
        z = m(xb); l = F.cross_entropy(z, yb); o.zero_grad(); l.backward(); o.step()
        sp += l.item() * len(k); ac += (z.argmax(1) == yb).sum().item()
    m.eval(); vp = va_ac = 0
    with torch.no_grad():
        for i in range(0, len(va), 256):
            k = va[i:i+256]; xb, yb = X[k].to(d), Y[k].to(d); z = m(xb)
            vp += F.cross_entropy(z, yb, reduction="sum").item(); va_ac += (z.argmax(1) == yb).sum().item()
    a = va_ac / len(va)
    H_["perda_treino"].append(sp / len(tr)); H_["acc_treino"].append(ac / len(tr)); H_["perda_val"].append(vp / len(va)); H_["acc_val"].append(a)
    if a > melhor[0]:
        melhor = (a, ep); torch.save({k: v.cpu() for k, v in m.state_dict().items()}, f"res/{nome}.pt")
    if sp != sp: break
json.dump(dict(nome=nome, cfg=cfg, flip=flip, seed=seed, batch=B, lr=lr, melhor_acc=melhor[0], melhor_epoca=melhor[1], segundos=time.time()-t0, hist=H_), open(f"res/{nome}.json", "w"))
print(f"RESULTADO {nome} | melhor val {melhor[0]:.3f} (ép {melhor[1]}) | final {a:.3f} | perda treino {H_['perda_treino'][-1]:.3f} | {ep} ép | {time.time()-t0:.0f}s", flush=True)
