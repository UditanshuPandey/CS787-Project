#!/usr/bin/env python3
# advgan_semi_whitebox_cifar10_resnet32_amp.py
# Semi-whitebox AdvGAN on CIFAR-10 using ONLY the paper-style ResNet-32 (6n+2 with n=5).
# - tqdm progress bars
# - AMP compatibility shim (works with torch.cuda.amp or torch.amp)
# - Saves: clean/adv/diff grids, sample_predictions.csv, ASR CSV, and a final JSON/CSV summary.

import os
import time
import json
import csv
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ============================================================
# AMP compatibility shim (handles both torch.cuda.amp and torch.amp)
# ============================================================

try:
    # New API (PyTorch ≥ 2.0 style)
    from torch import amp as _amp_mod

    def amp_autocast(enabled: bool = True):
        # torch.amp.autocast("cuda", ...)
        return _amp_mod.autocast("cuda", dtype=torch.float16, enabled=(enabled and torch.cuda.is_available()))

    class AMPGradScaler(_amp_mod.GradScaler):
        def __init__(self, enabled: bool = True):
            super().__init__("cuda", enabled=enabled and torch.cuda.is_available())

except Exception:
    # Old API fallback (torch.cuda.amp)
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScaler_old

    def amp_autocast(enabled: bool = True):
        # old signature: autocast(dtype=..., enabled=...)
        return _autocast_old(dtype=torch.float16, enabled=(enabled and torch.cuda.is_available()))

    class AMPGradScaler(_GradScaler_old):
        def __init__(self, enabled: bool = True):
            super().__init__(enabled=enabled and torch.cuda.is_available())

# ============================================================
# CIFAR-10 ResNet-32 (6n+2 with n=5)
# ============================================================

def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class CIFARBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)

class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks: List[int], num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = _conv3x3(3, 16, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(64*block.expansion, num_classes)
        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)

def resnet32_cifar(num_classes=10):
    # depth = 6n+2 -> n=5 -> [5,5,5]
    return ResNet_CIFAR(CIFARBasicBlock, [5,5,5], num_classes=num_classes)

# ============================================================
# AdvGAN generator/discriminator for RGB 32x32
# ============================================================

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            norm_layer(dim)
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, gen_input_nc=3, image_nc=3, nf=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(gen_input_nc, nf, 3, 1, 1, bias=True),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf*2, 3, 2, 1, bias=True),   # 32->16
            nn.InstanceNorm2d(nf*2),
            nn.ReLU(True),
            nn.Conv2d(nf*2, nf*4, 3, 2, 1, bias=True), # 16->8
            nn.InstanceNorm2d(nf*4),
            nn.ReLU(True),
        )
        self.bottleneck = nn.Sequential(
            ResnetBlock(nf*4), ResnetBlock(nf*4), ResnetBlock(nf*4), ResnetBlock(nf*4)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(nf*4, nf*2, 3, 2, 1, 1, bias=False), # 8->16
            nn.InstanceNorm2d(nf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf*2, nf, 3, 2, 1, 1, bias=False),   # 16->32
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, image_nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x): return self.decoder(self.bottleneck(self.encoder(x)))

class Discriminator(nn.Module):
    def __init__(self, image_nc=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_nc, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),   # 32->16
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True), # 16->8
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),# 8->4
            nn.Conv2d(256, 1, 4), nn.Sigmoid()                            # 4->1
        )
    def forward(self, x): return self.model(x).view(-1)

# ============================================================
# Helpers
# ============================================================

def weights_init_normal(m):
    n = m.__class__.__name__
    if 'Conv' in n or 'Linear' in n:
        try: nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception: pass
    if 'BatchNorm' in n or 'InstanceNorm' in n:
        try: nn.init.normal_(m.weight.data, 1.0, 0.02); nn.init.constant_(m.bias.data, 0)
        except Exception: pass

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def cw_loss(logits, labels, targeted=False, kappa=0.0):
    num_classes = logits.size(1)
    onehot = F.one_hot(labels, num_classes=num_classes).float().to(logits.device)
    real = torch.sum(onehot * logits, dim=1)
    other = torch.max((1 - onehot) * logits - (onehot * 1e4), dim=1).values
    if targeted:
        f = torch.clamp(other - real + kappa, min=0.0)
    else:
        f = torch.clamp(real - other + kappa, min=0.0)
    return f.mean()

def test_accuracy(model, dataloader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x,y in tqdm(dataloader, desc="Eval (acc)", leave=False):
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1,total)

def evaluate_asr(model, dataloader, attack_fn, device, only_on_correct=True, desc="Eval (ASR)"):
    model.eval(); success = 0; total = 0
    with torch.no_grad():
        for x,y in tqdm(dataloader, desc=desc, leave=False):
            x,y = x.to(device), y.to(device)
            if only_on_correct:
                cp = model(x).argmax(1)
                mask = cp.eq(y)
                if mask.sum().item() == 0: continue
                x, y = x[mask], y[mask]
            adv = attack_fn(x)
            ap = model(adv).argmax(1)
            success += (ap != y).sum().item()
            total += y.size(0)
    return success / max(1,total)

# ============================================================
# AdvGAN (semi-whitebox; frozen target)
# ============================================================

class AdvGAN_Attack:
    def __init__(self, device, model, image_nc=3, box_min=0.0, box_max=1.0,
                 epsilon=8/255.0, adv_kappa: float = 0.0, lr=2e-4):
        self.device = device
        self.model = model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.image_nc = image_nc
        self.box_min, self.box_max = box_min, box_max
        self.epsilon = epsilon
        self.adv_kappa = adv_kappa

        self.netG = Generator(image_nc, image_nc).to(device)
        self.netD = Discriminator(image_nc).to(device)
        self.netG.apply(weights_init_normal); self.netD.apply(weights_init_normal)

        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5,0.999))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5,0.999))

    def train(self, dataloader, epochs=20, save_dir: Optional[Path]=None):
        logs = []
        scalerG = AMPGradScaler(enabled=True)
        scalerD = AMPGradScaler(enabled=True)

        for ep in range(1, epochs+1):
            sums = {'D':0.0,'GAN':0.0,'PERT':0.0,'ADV':0.0}; n=0
            pbar = tqdm(dataloader, desc=f"AdvGAN epoch {ep}/{epochs}", leave=False)
            for x,y in pbar:
                x, y = x.to(self.device), y.to(self.device)

                # ----- D step (AMP) -----
                self.optimD.zero_grad(set_to_none=True)
                with amp_autocast(True):
                    raw = self.netG(x)                   # [-1,1]
                    perturb = self.epsilon * raw
                    adv = torch.clamp(x + perturb, self.box_min, self.box_max)
                    d_real = self.netD(x)
                    d_fake = self.netD(adv.detach())
                    lossD = 0.5*(F.mse_loss(d_real, torch.ones_like(d_real)) +
                                 F.mse_loss(d_fake, torch.zeros_like(d_fake)))
                scalerD.scale(lossD).backward()
                scalerD.step(self.optimD)
                scalerD.update()

                # ----- G step (AMP) -----
                self.optimG.zero_grad(set_to_none=True)
                with amp_autocast(True):
                    gan = F.mse_loss(self.netD(adv), torch.ones_like(d_fake))
                    pert_l2 = torch.mean(torch.norm(perturb.view(perturb.size(0), -1), p=2, dim=1))
                    logits = self.model(adv)
                    adv_loss = cw_loss(logits, y, targeted=False, kappa=self.adv_kappa)
                    lossG = adv_loss + 0.8*pert_l2 + 0.2*gan
                scalerG.scale(lossG).backward()
                scalerG.step(self.optimG)
                scalerG.update()

                sums['D']+=float(lossD.item()); sums['GAN']+=float(gan.item())
                sums['PERT']+=float(pert_l2.item()); sums['ADV']+=float(adv_loss.item()); n+=1
                pbar.set_postfix(D=f"{lossD.item():.3f}", GAN=f"{gan.item():.3f}",
                                 PERT=f"{pert_l2.item():.3f}", ADV=f"{adv_loss.item():.3f}")

            log = {'epoch':ep, 'loss_D':sums['D']/n, 'loss_GAN':sums['GAN']/n,
                   'loss_perturb':sums['PERT']/n, 'loss_adv':sums['ADV']/n}
            logs.append(log)
            tqdm.write(f"[AdvGAN] Epoch {ep}: D {log['loss_D']:.4f} | GAN {log['loss_GAN']:.4f} | "
                       f"PERT {log['loss_perturb']:.4f} | ADV {log['loss_adv']:.4f}")

            if save_dir is not None and (ep % 5 == 0 or ep == epochs):
                ensure_dir(save_dir)
                torch.save({'netG': self.netG.state_dict(), 'netD': self.netD.state_dict(),
                            'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()},
                           save_dir / f"advgan_epoch_{ep}.pt")
                save_json(logs, save_dir / "advgan_logs.json")
        return logs

    def generate(self, x):
        self.netG.eval()
        with torch.no_grad():
            raw = self.netG(x.to(self.device))
            perturb = self.epsilon * raw
            adv = torch.clamp(x.to(self.device) + perturb, self.box_min, self.box_max)
        return adv

# ============================================================
# Training & artifact helpers
# ============================================================

def train_classifier(model, train_loader, test_loader, device, epochs=30, lr=0.1, use_sgd=True):
    model = model.to(device)
    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2, int(0.75*epochs)], gamma=0.1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None

    ce = nn.CrossEntropyLoss()
    scaler = AMPGradScaler(enabled=True)

    for ep in range(1, epochs+1):
        model.train(); running=0.0
        pbar = tqdm(train_loader, desc=f"[ResNet32] epoch {ep}/{epochs}", leave=False)
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(True):
                out = model(x)
                loss = ce(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        if scheduler: scheduler.step()
        acc = test_accuracy(model, test_loader, device)
        tqdm.write(f"[Classifier] Epoch {ep}/{epochs} | Loss {running/len(train_loader):.4f} | Test Acc {acc:.4f}")
    return model

def save_sample_grids_and_csv(model, dataloader, gen_fn, outdir: Path, device):
    ensure_dir(outdir)
    imgs, labels = next(iter(dataloader))
    imgs, labels = imgs[:16].to(device), labels[:16].to(device)
    with torch.no_grad():
        advs = gen_fn(imgs)
        diffs = advs - imgs
        dmin, dmax = diffs.min(), diffs.max()
        diffs_norm = (diffs - dmin) / (dmax - dmin + 1e-12)
        vutils.save_image(imgs, str(outdir / "clean_grid.png"), normalize=True)
        vutils.save_image(advs, str(outdir / "adv_grid.png"), normalize=True)
        vutils.save_image(diffs_norm, str(outdir / "diff_grid.png"), normalize=False)
        clean_preds = model(imgs).argmax(1)
        adv_preds = model(advs).argmax(1)
    with open(outdir / "sample_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["index","label","clean_pred","adv_pred"])
        for i in range(imgs.size(0)):
            w.writerow([i, int(labels[i].cpu()), int(clean_preds[i].cpu()), int(adv_preds[i].cpu())])

# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = Path(f"./checkpoints/advgan_cifar10_resnet32_{timestamp}")
    ensure_dir(run_root)
    print("Artifacts ->", run_root)

    # Data (kept in [0,1] to match GAN expectations)
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test  = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10("./dataset_cifar10", train=True, download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10("./dataset_cifar10", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Build and train ResNet-32
    model_name = "ResNet32"
    model = resnet32_cifar(num_classes=10)
    tqdm.write(f"\n=== Training target: {model_name} ===")
    model = train_classifier(model, train_loader, test_loader, device, epochs=30, lr=0.1, use_sgd=True)
    acc = test_accuracy(model, test_loader, device)
    tqdm.write(f"[{model_name}] Test accuracy: {acc:.4f}")
    mdir = run_root / f"model_{model_name}"; ensure_dir(mdir)
    torch.save({"state": model.state_dict()}, mdir / f"{model_name}_final.pt")

    # Semi-whitebox AdvGAN
    tqdm.write("\n" + "="*70)
    tqdm.write(f"Semi-whitebox AdvGAN on {model_name}")
    tqdm.write("="*70)
    attack_dir = run_root / f"semi_whitebox_{model_name}"; ensure_dir(attack_dir)

    adv = AdvGAN_Attack(
        device=device, model=model, image_nc=3,
        box_min=0.0, box_max=1.0, epsilon=8/255.0, adv_kappa=0.0, lr=2e-4
    )
    adv_logs = adv.train(train_loader, epochs=30, save_dir=attack_dir)
    save_json(adv_logs, attack_dir / "advgan_train_logs.json")

    # save final G/D
    torch.save({
        "netG": adv.netG.state_dict(),
        "netD": adv.netD.state_dict(),
        "optimG": adv.optimG.state_dict(),
        "optimD": adv.optimD.state_dict()
    }, attack_dir / "advgan_final.pt")

    # Evaluate ASR
    attack_fn = lambda x: adv.generate(x)
    asr_only = evaluate_asr(model, test_loader, attack_fn, device, only_on_correct=True,  desc=f"ASR only-correct ({model_name})")
    asr_all  = evaluate_asr(model, test_loader, attack_fn, device, only_on_correct=False, desc=f"ASR all ({model_name})")
    tqdm.write(f"[{model_name}] ASR only-correct: {asr_only:.4f} | ASR all: {asr_all:.4f}")

    # Save sample grids + CSV
    save_sample_grids_and_csv(model, test_loader, attack_fn, attack_dir, device)

    # Per-target CSV
    with open(run_root / f"asr_summary_{model_name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","accuracy","asr_only_correct","asr_all","attack_dir"])
        w.writeheader()
        w.writerow({
            "model": model_name,
            "accuracy": float(acc),
            "asr_only_correct": float(asr_only),
            "asr_all": float(asr_all),
            "attack_dir": str(attack_dir)
        })

    # Top-level JSON + master CSV (single row)
    final = {"run_root": str(run_root),
             "model": {"name": model_name, "accuracy": float(acc), "checkpoint": str(mdir / f"{model_name}_final.pt")},
             "results": {"asr_only_correct": float(asr_only), "asr_all": float(asr_all), "attack_dir": str(attack_dir)}}
    save_json(final, run_root / "final_results.json")

    master_csv = run_root / "asr_summary_all_targets.csv"
    with open(master_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","accuracy","asr_only_correct","asr_all","attack_dir"])
        w.writeheader()
        w.writerow({"model": model_name, "accuracy": float(acc),
                    "asr_only_correct": float(asr_only), "asr_all": float(asr_all),
                    "attack_dir": str(attack_dir)})

    tqdm.write("\nAll done.")
    tqdm.write(f"Master CSV: {master_csv}")
    tqdm.write(f"Artifacts dir: {run_root}")

if __name__ == "__main__":
    main()
