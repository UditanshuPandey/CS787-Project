#!/usr/bin/env python3
# blackbox_dynamic_all_three.py
# Run dynamic distillation (black-box AdvGAN) for ALL THREE black-box targets: A, B, C.
# For each target: trains target, runs dynamic distillation with a substitute, saves grids and CSVs,
# and writes a master CSV summarizing ASRs across targets.

import json
import time
from pathlib import Path
import copy
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from typing import Optional

# ==================== MODELS ====================

class MNIST_target_net(nn.Module):
    # Model A
    def __init__(self):
        super(MNIST_target_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 200)
        self.fc2 = nn.Linear(200, 200)
        self.logits = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x)); x = F.relu(self.conv4(x)); x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x)); x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc2(x)); x = self.logits(x)
        return x

class MNIST_model_B(nn.Module):
    # Model B
    def __init__(self):
        super(MNIST_model_B, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(128*7*7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
        x = x.view(-1, 128*7*7)
        x = F.relu(self.fc1(x)); x = self.dropout(x); x = self.fc2(x)
        return x

class MNIST_model_C(nn.Module):
    # Model C (also used as default substitute)
    def __init__(self):
        super(MNIST_model_C, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x)); x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7); x = F.relu(self.fc1(x)); x = self.fc2(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, nf):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(nf)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + r

class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_nc, 8, 4, 2, 1, bias=True), nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, 4, 2, 1, bias=True), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1, bias=True), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.model(x).view(-1)

class Generator(nn.Module):
    def __init__(self, gen_input_nc, image_nc):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(gen_input_nc, 8, 3, 1, 1, bias=True), nn.InstanceNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1, bias=True), nn.InstanceNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1, bias=True), nn.InstanceNorm2d(32), nn.ReLU(),
        )
        self.bottle_neck = nn.Sequential(ResnetBlock(32), ResnetBlock(32), ResnetBlock(32), ResnetBlock(32))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1, bias=False), nn.InstanceNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, 2, 1, 1, bias=False), nn.InstanceNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, image_nc, 3, 1, 1, bias=False), nn.Tanh()
        )
    def forward(self, x): return self.decoder(self.bottle_neck(self.encoder(x)))

# ==================== AdvGAN (paper-ish losses) ====================

class AdvGAN_PaperExact:
    def __init__(self, device, model, model_num_labels, image_nc, box_min, box_max,
                 epsilon=0.3, c=2.0, alpha=1.0, beta=1.0):
        self.device = device
        self.model = model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.box_min, self.box_max = box_min, box_max
        self.netG = Generator(image_nc, image_nc).to(device)
        self.netDisc = Discriminator(image_nc).to(device)
        self.netG.apply(self._init); self.netDisc.apply(self._init)
        self.optG = torch.optim.Adam(self.netG.parameters(), lr=2e-4)
        self.optD = torch.optim.Adam(self.netDisc.parameters(), lr=2e-4)
        self.c, self.alpha, self.beta = c, alpha, beta

    def _init(self, m):
        n = m.__class__.__name__
        if 'Conv' in n or 'Linear' in n:
            try: nn.init.normal_(m.weight.data, 0.0, 0.02)
            except Exception: pass
        if 'BatchNorm' in n or 'InstanceNorm' in n:
            try: nn.init.normal_(m.weight.data, 1.0, 0.02); nn.init.constant_(m.bias.data, 0)
            except Exception: pass

    def _cw_loss(self, logits, t, kappa=0.0):
        onehot = torch.zeros_like(logits); onehot.scatter_(1, t.unsqueeze(1), 1)
        real = (onehot * logits).sum(1)
        other = ((1 - onehot) * logits - onehot * 1e4).max(1)[0]
        return torch.clamp(other - real + kappa, min=0.0).mean()

    def train(self, dl, epochs=3, target_class=None):
        logs = []
        for ep in range(1, epochs+1):
            sums = {k:0.0 for k in ['D','G','ADV','HINGE','GAN']}
            n = 0
            for x, y in dl:
                x, y = x.to(self.device), y.to(self.device)
                t = torch.full_like(y, target_class) if target_class is not None else (y + torch.randint(1,10,y.shape, device=y.device))%10

                # D
                self.optD.zero_grad()
                p = self.netG(x)
                adv = torch.clamp(x + p, 0.0, 1.0)
                d_real = self.netDisc(x); d_fake = self.netDisc(adv.detach())
                lossD = 0.5*(F.binary_cross_entropy(d_real, torch.ones_like(d_real))
                            +F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake)))
                lossD.backward(); self.optD.step()

                # G
                self.optG.zero_grad()
                gan = F.binary_cross_entropy(self.netDisc(adv), torch.ones_like(d_fake))
                adv_logits = self.model(adv)
                adv_loss = self._cw_loss(adv_logits, t, 0.0)
                hinge = torch.clamp(torch.norm(p.view(p.size(0), -1), p=2, dim=1) - self.c, min=0.0).mean()
                lossG = adv_loss + self.alpha*gan + self.beta*hinge
                lossG.backward(); self.optG.step()

                sums['D']+=lossD.item(); sums['G']+=lossG.item(); sums['ADV']+=adv_loss.item(); sums['HINGE']+=hinge.item(); sums['GAN']+=gan.item(); n+=1
            log = {'epoch':ep, 'loss_D':sums['D']/n, 'loss_G':sums['G']/n, 'loss_adv':sums['ADV']/n, 'loss_hinge':sums['HINGE']/n, 'loss_gan':sums['GAN']/n}
            logs.append(log)
            print(f"Epoch {ep}: D {log['loss_D']:.3f} | G {log['loss_G']:.3f} | ADV {log['loss_adv']:.3f} | H {log['loss_hinge']:.3f}")
        return logs

    def generate_adv_examples(self, x):
        self.netG.eval()
        with torch.no_grad():
            p = self.netG(x)
            adv = torch.clamp(x + p, 0.0, 1.0)
        self.netG.train()
        return adv

# ==================== Dynamic Distillation ====================
class DynamicDistiller:
    def __init__(self, blackbox_model, substitute_model_class, device, image_nc=1, box_min=0.0, box_max=1.0, epsilon=0.3):
        self.bb = blackbox_model.eval()
        for p in self.bb.parameters(): p.requires_grad_(False)
        self.sub = substitute_model_class().to(device)
        self.device = device
        self.image_nc = image_nc
        self.box_min, self.box_max = box_min, box_max
        self.epsilon = epsilon
        self.advgan = None

    def _static_distill(self, dl, epochs=5):
        opt = torch.optim.Adam(self.sub.parameters(), lr=1e-3)
        self.sub.train()
        for ep in range(epochs):
            tot, n = 0.0, 0
            for x, _ in dl:
                x = x.to(self.device)
                with torch.no_grad():
                    ybb = self.bb(x).argmax(1)
                opt.zero_grad()
                loss = F.cross_entropy(self.sub(x), ybb)
                loss.backward(); opt.step()
                tot += loss.item(); n += 1
            if ep % 2 == 0: print(f"  Static distill epoch {ep}/{epochs}: {tot/max(n,1):.4f}")

    def _collect(self, dl, max_samples=20000):
        imgs = []
        for x, _ in dl:
            imgs.append(x)
            if len(torch.cat(imgs,0)) >= max_samples: break
        imgs = torch.cat(imgs,0)[:max_samples]
        labels = []
        with torch.no_grad():
            for i in range(0, len(imgs), 256):
                batch = imgs[i:i+256].to(self.device)
                labels.append(self.bb(batch).argmax(1).cpu())
        labels = torch.cat(labels,0)[:max_samples]
        return imgs, labels

    def _gen_adv_and_query(self, imgs, num_samples=2000):
        idx = torch.randperm(len(imgs))[:num_samples]
        sub = imgs[idx]
        with torch.no_grad():
            adv = self.advgan.generate_adv_examples(sub.to(self.device))
        preds = []
        with torch.no_grad():
            for i in range(0, len(adv), 256):
                batch = adv[i:i+256]
                preds.append(self.bb(batch).argmax(1).cpu())
        return adv.cpu(), torch.cat(preds,0)

    def _train_sub_combined(self, clean_x, clean_y, adv_x, adv_y, epochs=3):
        for p in self.sub.parameters(): p.requires_grad_(True)
        opt = torch.optim.Adam(self.sub.parameters(), lr=1e-3)
        ds = torch.utils.data.TensorDataset(torch.cat([clean_x, adv_x]), torch.cat([clean_y, adv_y]))
        dl = DataLoader(ds, batch_size=128, shuffle=True)
        self.sub.train()
        for ep in range(epochs):
            tot, n = 0.0, 0
            for x, y in dl:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = F.cross_entropy(self.sub(x), y)
                loss.backward(); opt.step()
                tot += loss.item(); n += 1
            print(f"    Distill epoch {ep}/{epochs}: {tot/max(n,1):.4f}")

    def run(self, initial_dl, iters=3, adv_epochs=3, distill_epochs=2, target_class=None, save_dir: Optional[Path]=None):
        if save_dir: save_dir.mkdir(parents=True, exist_ok=True)
        print("Initial static distillation...")
        self._static_distill(initial_dl, epochs=5)
        clean_x, clean_y = self._collect(initial_dl)
        logs = []
        for it in range(1, iters+1):
            print(f"\n=== Iteration {it}/{iters} ===")
            frozen = copy.deepcopy(self.sub).to(self.device).eval()
            for p in frozen.parameters(): p.requires_grad_(False)
            self.advgan = AdvGAN_PaperExact(self.device, frozen, 10, self.image_nc, self.box_min, self.box_max, epsilon=self.epsilon)
            adv_logs = self.advgan.train(initial_dl, epochs=adv_epochs, target_class=target_class)
            adv_x, adv_y = self._gen_adv_and_query(clean_x)
            self._train_sub_combined(clean_x, clean_y, adv_x, adv_y, epochs=distill_epochs)
            # eval quick metrics
            acc = evaluate_accuracy(self.sub, clean_x[:1000], clean_y[:1000], self.device)
            agr = measure_agreement(self.sub, self.bb, clean_x[:1000], self.device)
            print(f"  Substitute acc: {acc:.4f} | Agreement: {agr:.4f}")
            entry = {'iteration': it, 'sub_acc': float(acc), 'agreement': float(agr), 'adv_logs': adv_logs}
            logs.append(entry)
            if save_dir:
                itdir = save_dir / f"iter_{it}"; itdir.mkdir(parents=True, exist_ok=True)
                torch.save({'sub': self.sub.state_dict(),
                            'G': self.advgan.netG.state_dict(),
                            'D': self.advgan.netDisc.state_dict()}, itdir / "snapshot.pt")
                with open(itdir / "adv_logs.json", "w") as f: json.dump(adv_logs, f, indent=2)
        return logs, self.advgan

# ==================== EVAL & UTILS ====================

def evaluate_asr(model, dataloader, attack_method, device, only_on_correct=True):
    model.eval()
    total = 0; success = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if only_on_correct:
                cp = model(x).argmax(1); m = cp.eq(y)
                if m.sum() == 0: continue
                x, y = x[m], y[m]
            adv = attack_method(x)
            ap = model(adv).argmax(1)
            success += (ap != y).sum().item()
            total += y.size(0)
    return success / max(total, 1)

def evaluate_accuracy(model, images, labels, device):
    model.eval()
    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        return (model(images).argmax(1) == labels).float().mean().item()

def measure_agreement(sub, bb, images, device):
    sub.eval(); bb.eval()
    with torch.no_grad():
        images = images.to(device)
        return (sub(images).argmax(1) == bb(images).argmax(1)).float().mean().item()

def test_model_accuracy(model, dataloader, device):
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            total += y.size(0); correct += (model(x).argmax(1) == y).sum().item()
    return correct / max(total, 1)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def save_checkpoint(obj, path: Path):
    torch.save(obj, str(path))

def save_grids_and_csv(eval_model, dataloader, gen_fn, outdir: Path, device):
    ensure_dir(outdir)
    x, y = next(iter(dataloader))
    x, y = x[:16].to(device), y[:16].to(device)
    with torch.no_grad():
        adv = gen_fn(x)
        diff = adv - x
        dmin, dmax = diff.min(), diff.max()
        diffn = (diff - dmin) / (dmax - dmin + 1e-12)
        vutils.save_image(x, str(outdir / "clean_grid.png"), normalize=True)
        vutils.save_image(adv, str(outdir / "adv_grid.png"), normalize=True)
        vutils.save_image(diffn, str(outdir / "diff_grid.png"), normalize=False)
        cp = eval_model(x).argmax(1); ap = eval_model(adv).argmax(1)
    with open(outdir / "sample_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["index","clean_pred","adv_pred","label"])
        for i in range(x.size(0)):
            w.writerow([i, int(cp[i].cpu()), int(ap[i].cpu()), int(y[i].cpu())])

# ==================== MAIN (ALL THREE BLACK-BOX TARGETS) ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(f"./checkpoints/blackbox_dynamic_all_{timestamp}")
    ensure_dir(root)

    batch_size = 128
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST("./dataset", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST("./dataset", train=False, download=True, transform=transform)
    train_dl = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Map model names to classes for black-box target training
    model_map = {
        "A": MNIST_target_net,
        "B": MNIST_model_B,
        "C": MNIST_model_C
    }

    # Master CSV
    master_csv = root / "asr_summary_all_targets.csv"
    with open(master_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blackbox_model","blackbox_accuracy",
                                          "substitute_accuracy","substitute_asr_only_correct","substitute_asr_all",
                                          "blackbox_asr_only_correct","blackbox_asr_all",
                                          "run_dir"])
        w.writeheader()

    for name, cls in model_map.items():
        print("\n" + "="*80)
        print(f"BLACK-BOX TARGET: Model {name}")
        print("="*80)
        run_dir = root / f"target_{name}"
        ensure_dir(run_dir)

        # Train black-box model (as target)
        bb = cls().to(device)
        opt = torch.optim.Adam(bb.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        bb.train()
        for ep in range(5):
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(); loss = crit(bb(x), y); loss.backward(); opt.step()
        bb_acc = test_model_accuracy(bb, test_dl, device)
        print(f"  Trained black-box Model {name} accuracy: {bb_acc:.4f}")
        save_checkpoint({"state": bb.state_dict()}, run_dir / f"blackbox_model_{name}.pt")

        # Dynamic distillation (use Model C as substitute by default)
        dyn_dir = run_dir / "dynamic"
        distiller = DynamicDistiller(blackbox_model=bb, substitute_model_class=MNIST_model_C,
                                     device=device, image_nc=1, box_min=0.0, box_max=1.0, epsilon=0.3)
        logs, advgan = distiller.run(initial_dl=train_dl, iters=3, adv_epochs=3, distill_epochs=2,
                                     target_class=None, save_dir=dyn_dir)
        save_json(logs, dyn_dir / "dynamic_logs.json")

        # Evaluate & save artifacts
        def gen_fn(t): return advgan.generate_adv_examples(t)

        sub_dir = run_dir / "substitute_eval"; ensure_dir(sub_dir)
        sub_acc = test_model_accuracy(distiller.sub, test_dl, device)
        sub_asr_correct = evaluate_asr(distiller.sub, test_dl, gen_fn, device, only_on_correct=True)
        sub_asr_all = evaluate_asr(distiller.sub, test_dl, gen_fn, device, only_on_correct=False)
        save_grids_and_csv(distiller.sub, test_dl, gen_fn, sub_dir, device)

        bb_dir = run_dir / "blackbox_eval"; ensure_dir(bb_dir)
        bb_asr_correct = evaluate_asr(bb, test_dl, gen_fn, device, only_on_correct=True)
        bb_asr_all = evaluate_asr(bb, test_dl, gen_fn, device, only_on_correct=False)
        save_grids_and_csv(bb, test_dl, gen_fn, bb_dir, device)

        # Save per-target summary CSV
        with open(run_dir / "asr_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model","accuracy","asr_only_correct","asr_all","eval_dir"])
            w.writeheader()
            w.writerow({"model":"substitute","accuracy":sub_acc,"asr_only_correct":sub_asr_correct,
                        "asr_all":sub_asr_all,"eval_dir":str(sub_dir)})
            w.writerow({"model":f"blackbox_{name}","accuracy":bb_acc,"asr_only_correct":bb_asr_correct,
                        "asr_all":bb_asr_all,"eval_dir":str(bb_dir)})

        # Save checkpoints
        save_checkpoint({
            "blackbox_model": bb.state_dict(),
            "substitute_model": distiller.sub.state_dict(),
            "advG": advgan.netG.state_dict(),
            "advD": advgan.netDisc.state_dict()
        }, run_dir / "final_models.pt")

        # Save final JSON
        final = {
            "blackbox": {"name": name, "accuracy": float(bb_acc)},
            "substitute": {"accuracy": float(sub_acc)},
            "ASR": {
                "substitute_only_correct": float(sub_asr_correct),
                "substitute_all": float(sub_asr_all),
                "blackbox_only_correct": float(bb_asr_correct),
                "blackbox_all": float(bb_asr_all)
            },
            "artifacts": {
                "dynamic_dir": str(dyn_dir),
                "substitute_eval_dir": str(sub_dir),
                "blackbox_eval_dir": str(bb_dir),
                "per_target_csv": str(run_dir / "asr_summary.csv")
            },
            "logs": logs
        }
        save_json(final, run_dir / "final_results.json")

        # Append to master CSV
        with open(master_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["blackbox_model","blackbox_accuracy",
                                              "substitute_accuracy","substitute_asr_only_correct","substitute_asr_all",
                                              "blackbox_asr_only_correct","blackbox_asr_all",
                                              "run_dir"])
            w.writerow({
                "blackbox_model": name,
                "blackbox_accuracy": bb_acc,
                "substitute_accuracy": sub_acc,
                "substitute_asr_only_correct": sub_asr_correct,
                "substitute_asr_all": sub_asr_all,
                "blackbox_asr_only_correct": bb_asr_correct,
                "blackbox_asr_all": bb_asr_all,
                "run_dir": str(run_dir)
            })

        print(f"\nSaved artifacts for target {name} in: {run_dir}")

    print("\nAll targets complete.")
    print(f"Master ASR CSV: {master_csv}")
    return str(root)

if __name__ == "__main__":
    main()