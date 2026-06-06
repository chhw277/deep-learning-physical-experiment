"""快速评估：19组数据 PSNR/SSIM"""
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from polarized_csi.model_pt import DL_M1

device = torch.device('cpu')
model = DL_M1().to(device)
model.load_state_dict(torch.load('dl_m1.pt', map_location=device, weights_only=True))
model.eval()

comp_dir = Path(r'D:\courser workshop\My rearch math\compressed')
stokes_dir = Path(r'D:\courser workshop\My rearch math\stokes_npy')

def psnr(pred, target):
    mse = np.mean((pred - target)**2)
    if mse < 1e-10:
        return 100.0
    dr = target.max() - target.min()
    if dr < 1e-10:
        return 100.0
    return 10*np.log10(dr**2/mse)

def ssim_ch(pred, target):
    dr = target.max() - target.min()
    if dr < 1e-10:
        return 1.0
    return float(ssim_fn(target, pred, data_range=dr))

header = f"{'Group':<16} {'PSNR_S0':>8} {'PSNR_S1':>8} {'PSNR_S2':>8} {'PSNR_S3':>8} {'mSSIM':>8}"
print(header)
print('-'*len(header))
all_ssim = []
all_psnr = []

for npy in sorted(stokes_dir.glob('*.npy')):
    name = npy.stem
    cp = None
    for ext in ['.bmp','.png','.jpg']:
        p = comp_dir/(name+ext)
        if p.exists():
            cp = p
            break
    if cp is None:
        continue
    comp = np.array(Image.open(cp).convert('L'), dtype=np.float32)/255.0
    gt = np.load(npy).astype(np.float32)
    x = torch.from_numpy(comp).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).squeeze(0).numpy()  # (4,H,W)
    gt_t = gt.transpose(2,0,1)
    psnrs = [psnr(pred[c], gt_t[c]) for c in range(4)]
    ssims = [ssim_ch(pred[c], gt_t[c]) for c in range(4)]
    ms = np.mean(ssims)
    all_ssim.append(ms)
    all_psnr.append(psnrs)
    print(f'{name:<16} {psnrs[0]:8.2f} {psnrs[1]:8.2f} {psnrs[2]:8.2f} {psnrs[3]:8.2f} {ms:8.4f}')

print('-'*len(header))
avg_psnr = np.mean(all_psnr, axis=0)
avg_ssim = np.mean(all_ssim)
print(f"{'AVERAGE':<16} {avg_psnr[0]:8.2f} {avg_psnr[1]:8.2f} {avg_psnr[2]:8.2f} {avg_psnr[3]:8.2f} {avg_ssim:8.4f}")
