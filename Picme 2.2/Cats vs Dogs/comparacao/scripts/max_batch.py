# Maior batch em que passos completos de treino cabem na GPU, de forma repetível, em cada configuração.
import torch, torch.nn.functional as F, gc
from comp_lib import rede, CFG
N = 21520; d = "cuda"
def cabe(cfg, B):
    H, W, p, c = CFG[cfg]; m = None
    try:
        m, _ = rede(H, W, p, c); m = m.to(d); o = torch.optim.SGD(m.parameters(), lr=1e-2)
        for _ in range(2):
            x = torch.rand(B, 3, H, W, device=d); y = torch.randint(0, 2, (B,), device=d)
            l = F.cross_entropy(m(x), y); o.zero_grad(); l.backward(); o.step()
        return True, torch.cuda.max_memory_allocated() / 2**30
    except RuntimeError:   # falta de memória, inclusive a que aparece como erro do cuDNN
        return False, None
    finally:
        m = x = y = l = o = None; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
def estavel(cfg, B, vezes=3):
    return all(cabe(cfg, B)[0] for _ in range(vezes))
def maior(cfg):
    if estavel(cfg, N): return N
    lo, hi = 32, N
    while hi - lo > max(16, lo // 50):
        mid = (lo + hi) // 2
        if estavel(cfg, mid): lo = mid
        else: hi = mid
    return lo
for cfg in CFG:
    B = maior(cfg); folga = min(N, int(B * .9)); ok, pico = cabe(cfg, folga); _, s = rede(*CFG[cfg])
    print(f"{cfg}: saída dos blocos {s} | maior estável {B} | com folga de 10%: {folga} (ok={ok}, pico {pico if pico is None else round(pico,1)} GiB) | passos por época {-(-N // folga)} (hoje: {-(-N // 32)})", flush=True)
