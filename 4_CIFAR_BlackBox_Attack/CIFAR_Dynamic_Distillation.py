import json
import time
import copy
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split

# -------------------------
# Simple ResNet-20 for substitute (6n+2 with n=3 -> [3,3,3])
# -------------------------

def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)
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
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = _conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(64*block.expansion, num_classes)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        out = self.avgpool(out); out = torch.flatten(out, 1)
        return self.linear(out)

def resnet20_cifar():
    return ResNet_CIFAR(BasicBlock, [3,3,3])

def resnet32_cifar():
    return ResNet_CIFAR(BasicBlock, [5,5,5])

# -------------------------
# AdvGAN generator & discriminator for CIFAR-10 (RGB 32x32)
# -------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, ngf=64, input_nc=3, output_nc=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 3, 1, 1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, 2, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True)
        )
        self.bottleneck = nn.Sequential(*[ResnetBlock(ngf*4) for _ in range(4)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, 1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, output_nc, 3, 1, 1), nn.Tanh()
        )
    def forward(self, x):
        return self.decoder(self.bottleneck(self.encoder(x)))

class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x).view(-1)

# -------------------------
# Helpers: saving, losses, eval
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def save_checkpoint(obj: dict, path: Path):
    torch.save(obj, str(path))

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
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total += y.size(0)
    return correct / max(1,total)

def evaluate_asr(model, dataloader, attack_fn, device, only_on_correct=True):
    model.eval(); succ=0; total=0
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device), y.to(device)
            if only_on_correct:
                cp = model(x).argmax(1); mask = cp.eq(y)
                if mask.sum().item() == 0: continue
                x,y = x[mask], y[mask]
            adv = attack_fn(x)
            p = model(adv).argmax(1)
            succ += (p != y).sum().item(); total += y.size(0)
    return succ / max(1,total)

# -------------------------
# AdvGAN (paper-faithful) class for CIFAR
# -------------------------
class AdvGAN_Paper:
    def __init__(self, device, target_model, image_nc=3, epsilon=8/255.0, c=2.0, alpha=1.0, beta=1.0, gen_lr=2e-4, disc_lr=2e-4):
        self.device = device
        self.target = target_model.eval()
        for p in self.target.parameters(): p.requires_grad_(False)
        self.image_nc = image_nc
        self.epsilon = epsilon
        self.c, self.alpha, self.beta = c, alpha, beta
        self.netG = Generator().to(device)
        self.netD = Discriminator().to(device)
        self.netG.apply(self._init_weights); self.netD.apply(self._init_weights)
        self.optG = torch.optim.Adam(self.netG.parameters(), lr=gen_lr, betas=(0.5,0.999))
        self.optD = torch.optim.Adam(self.netD.parameters(), lr=disc_lr, betas=(0.5,0.999))

    def _init_weights(self, m):
        name = m.__class__.__name__
        if 'Conv' in name or 'Linear' in name:
            try: nn.init.normal_(m.weight.data, 0.0, 0.02)
            except Exception: pass
        if 'BatchNorm' in name or 'InstanceNorm' in name:
            try: nn.init.normal_(m.weight.data, 1.0, 0.02); nn.init.constant_(m.bias.data, 0)
            except Exception: pass

    def train(self, dataloader, epochs=30, save_dir: Optional[Path]=None, checkpoint_every=5):
        ensure_dir(save_dir) if save_dir is not None else None
        logs = []
        for ep in range(1, epochs+1):
            self.netG.train(); self.netD.train()
            sums = {'D':0.0,'G':0.0,'ADV':0.0,'HINGE':0.0,'GAN':0.0}; n=0
            for x,y in dataloader:
                x,y = x.to(self.device), y.to(self.device)
                # choose target t randomly != y for targeted variant in paper sometimes; here we follow untargeted CW
                # ----- D -----
                p = self.netG(x)
                adv = torch.clamp(x + p, 0.0, 1.0)
                self.optD.zero_grad()
                d_real = self.netD(x); d_fake = self.netD(adv.detach())
                lossD = 0.5*(F.binary_cross_entropy(d_real, torch.ones_like(d_real)) + F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake)))
                lossD.backward(); self.optD.step()
                # ----- G -----
                self.optG.zero_grad()
                gan_loss = F.binary_cross_entropy(self.netD(adv), torch.ones_like(d_real))
                logits = self.target(adv)
                # use untargeted cw: maximize difference between true and other
                adv_loss = cw_loss(logits, y, targeted=False, kappa=0.0)
                hinge = torch.clamp(torch.norm(p.view(p.size(0), -1), p=2, dim=1) - self.c, min=0.0).mean()
                lossG = adv_loss + self.alpha * gan_loss + self.beta * hinge
                lossG.backward(); self.optG.step()

                sums['D'] += lossD.item(); sums['G'] += lossG.item(); sums['ADV'] += adv_loss.item(); sums['HINGE'] += hinge.item(); sums['GAN'] += gan_loss.item(); n += 1

            log = {'epoch': ep, 'loss_D': sums['D']/n, 'loss_G': sums['G']/n, 'loss_adv': sums['ADV']/n, 'loss_hinge': sums['HINGE']/n, 'loss_gan': sums['GAN']/n}
            logs.append(log)
            print(f"AdvGAN epoch {ep}/{epochs} | D {log['loss_D']:.4f} | G {log['loss_G']:.4f} | adv {log['loss_adv']:.4f}")

            if save_dir is not None and (ep % checkpoint_every == 0 or ep == epochs):
                ensure_dir(save_dir)
                save_checkpoint({'epoch':ep, 'netG':self.netG.state_dict(), 'netD':self.netD.state_dict(), 'optG':self.optG.state_dict(), 'optD':self.optD.state_dict()}, save_dir / f"checkpoint_epoch_{ep}.pt")
                save_json(logs, save_dir / "advgan_per_epoch_logs.json")
                save_json(logs, save_dir / "per_epoch_logs.json")
                save_checkpoint({'epoch':ep, 'netG':self.netG.state_dict(), 'netD':self.netD.state_dict()}, save_dir / "best_checkpoint.pt")

        if save_dir is not None:
            save_checkpoint({'netG':self.netG.state_dict(), 'netD':self.netD.state_dict()}, save_dir / "final_checkpoint.pt")
            save_json(logs, save_dir / "advgan_per_epoch_logs.json")
        return logs

    def generate(self, x):
        self.netG.eval()
        with torch.no_grad():
            p = self.netG(x)
            adv = torch.clamp(x + p, 0.0, 1.0)
        return adv

# -------------------------
# Dynamic distillation (paper-faithful)
# -------------------------
class DynamicDistillerCIFAR:
    def __init__(self, blackbox_model: nn.Module, substitute_class, device, image_nc=3, box_min=0.0, box_max=1.0, epsilon=8/255.0):
        self.bb = blackbox_model.eval()
        for p in self.bb.parameters(): p.requires_grad_(False)
        self.sub = substitute_class().to(device)
        self.device = device
        self.image_nc = image_nc
        self.box_min = box_min; self.box_max = box_max; self.epsilon = epsilon
        self.advgan: Optional[AdvGAN_Paper] = None

    def static_distill(self, dl, epochs=10, lr=1e-3):
        opt = torch.optim.Adam(self.sub.parameters(), lr=lr)
        self.sub.train()
        for ep in range(1, epochs+1):
            total=0.0; n=0
            for x, _ in dl:
                x = x.to(self.device)
                with torch.no_grad(): ybb = self.bb(x).argmax(1)
                opt.zero_grad(); loss = F.cross_entropy(self.sub(x), ybb); loss.backward(); opt.step()
                total += loss.item(); n += 1
            if ep % 2 == 0 or ep == epochs:
                print(f"  Static distill ep {ep}/{epochs}: loss {total/max(1,n):.4f}")

    def collect_pristine(self, dl, max_samples=20000):
        imgs = []
        for x, _ in dl:
            imgs.append(x)
            if sum(t.size(0) for t in imgs) >= max_samples:
                break
        imgs = torch.cat(imgs, 0)[:max_samples]
        labels = []
        with torch.no_grad():
            for i in range(0, len(imgs), 256):
                batch = imgs[i:i+256].to(self.device)
                labels.append(self.bb(batch).argmax(1).cpu())
        labels = torch.cat(labels, 0)[:len(imgs)]
        return imgs, labels

    def generate_adv_and_query(self, imgs, num_samples=2000):
        idx = torch.randperm(len(imgs))[:num_samples]
        sub = imgs[idx]
        adv = self.advgan.generate(sub.to(self.device)).cpu()
        preds = []
        with torch.no_grad():
            for i in range(0, len(adv), 256):
                batch = adv[i:i+256].to(self.device)
                preds.append(self.bb(batch).argmax(1).cpu())
        return adv, torch.cat(preds, 0)

    def update_substitute_with_adv(self, clean_x, clean_y, adv_x, adv_y, epochs=3, lr=1e-3):
        ds = torch.utils.data.TensorDataset(torch.cat([clean_x, adv_x]), torch.cat([clean_y, adv_y]))
        dl = DataLoader(ds, batch_size=128, shuffle=True)
        opt = torch.optim.Adam(self.sub.parameters(), lr=lr)
        self.sub.train()
        for ep in range(1, epochs+1):
            tot=0.0; n=0
            for x,y in dl:
                x,y = x.to(self.device), y.to(self.device)
                opt.zero_grad(); loss = F.cross_entropy(self.sub(x), y); loss.backward(); opt.step()
                tot += loss.item(); n += 1
            print(f"    Sub distill ep {ep}/{epochs}: loss {tot/max(1,n):.4f}")

    def run(self, train_dl, iters=3, adv_epochs=30, distill_epochs=3, static_epochs=10, save_dir: Optional[Path]=None):
        if save_dir: ensure_dir(save_dir)
        print("Starting static distillation...")
        self.static_distill(train_dl, epochs=static_epochs)
        pristine_x, pristine_y = self.collect_pristine(train_dl, max_samples=20000)
        logs = []
        for it in range(1, iters+1):
            print(f"\n=== Dynamic iteration {it}/{iters} ===")
            frozen = copy.deepcopy(self.sub).to(self.device).eval()
            for p in frozen.parameters(): p.requires_grad_(False)
            # train AdvGAN against frozen substitute
            adv_dir = save_dir / f"advgan_iter_{it}" if save_dir is not None else None
            self.advgan = AdvGAN_Paper(self.device, frozen, image_nc=self.image_nc, epsilon=self.epsilon)
            adv_logs = self.advgan.train(train_dl, epochs=adv_epochs, save_dir=adv_dir)
            # generate adversarials and query blackbox
            adv_x, adv_y = self.generate_adv_and_query(pristine_x, num_samples=2000)
            # update substitute
            self.update_substitute_with_adv(pristine_x, pristine_y, adv_x, adv_y, epochs=distill_epochs)
            # quick eval
            sub_acc = test_accuracy(self.sub, DataLoader(torch.utils.data.TensorDataset(pristine_x, pristine_y), batch_size=256), self.device)
            agreement = (self.sub(pristine_x.to(self.device)).argmax(1).cpu() == self.bb(pristine_x.to(self.device)).argmax(1).cpu()).float().mean().item()
            print(f"  Substitute acc: {sub_acc:.4f} | Agreement: {agreement:.4f}")
            entry = {'iteration': it, 'sub_acc': float(sub_acc), 'agreement': float(agreement), 'adv_logs': adv_logs}
            logs.append(entry)
            if save_dir:
                itdir = save_dir / f"iter_{it}"; ensure_dir(itdir)
                save_checkpoint({'sub': self.sub.state_dict(), 'advG': self.advgan.netG.state_dict(), 'advD': self.advgan.netD.state_dict()}, itdir / "snapshot.pt")
                save_json(adv_logs, itdir / "advgan_per_epoch_logs.json")
                save_json(entry, itdir / "iter_summary.json")
        return logs, self.advgan

# -------------------------
# Saving grids and CSV helpers
# -------------------------

def save_sample_grids_and_csv(eval_model, dataloader, gen_fn, outdir: Path, device):
    ensure_dir(outdir)
    imgs, labels = next(iter(dataloader)); imgs, labels = imgs[:16].to(device), labels[:16].to(device)
    with torch.no_grad():
        advs = gen_fn(imgs)
        diffs = advs - imgs
        dmin, dmax = diffs.min(), diffs.max()
        diffs_norm = (diffs - dmin) / (dmax - dmin + 1e-12)
        vutils.save_image(imgs, str(outdir / 'clean_grid.png'), normalize=True)
        vutils.save_image(advs, str(outdir / 'adv_grid.png'), normalize=True)
        vutils.save_image(diffs_norm, str(outdir / 'diff_grid.png'), normalize=False)
        clean_preds = eval_model(imgs).argmax(1); adv_preds = eval_model(advs).argmax(1)
    with open(outdir / 'sample_predictions.csv', 'w', newline='') as f:
        import csv
        w = csv.writer(f); w.writerow(['index','clean_pred','adv_pred','label'])
        for i in range(imgs.size(0)):
            w.writerow([i, int(clean_preds[i].cpu()), int(adv_preds[i].cpu()), int(labels[i].cpu())])

# -------------------------
# Main pipeline: iterate targets (A/B/C) on CIFAR-10
# -------------------------

def main():
    random.seed(0); torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    root = Path(f'./checkpoints/cifar10_blackbox_dynamic_{timestamp}')
    ensure_dir(root)

    batch_size = 128
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10('./dataset_cifar10', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10('./dataset_cifar10', train=False, download=True, transform=transform)

    # split train -> train/val for stability
    val_size = 5000; train_size = len(trainset) - val_size
    train_subset, val_subset = random_split(trainset, [train_size, val_size])
    train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dl = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model_map = {'A': resnet32_cifar, 'B': resnet32_cifar, 'C': resnet32_cifar}

    master_csv = root / 'asr_summary_all_targets.csv'
    with open(master_csv, 'w', newline='') as f:
        import csv
        w = csv.DictWriter(f, fieldnames=['blackbox_model','blackbox_accuracy','substitute_accuracy','substitute_asr_only_correct','substitute_asr_all','blackbox_asr_only_correct','blackbox_asr_all','run_dir'])
        w.writeheader()

    for name, cls in model_map.items():
        print('\n' + '='*80)
        print(f'TARGET: {name}')
        run_dir = root / f'target_{name}'; ensure_dir(run_dir)

        # Train black-box (ResNet32) - paper uses strong models; we train for 100 epochs or less depending on resources
        bb = cls().to(device)
        opt = torch.optim.SGD(bb.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 80], gamma=0.1)
        crit = nn.CrossEntropyLoss()
        epochs_bb = 100
        for ep in range(1, epochs_bb+1):
            bb.train(); total=0.0; n=0
            for x,y in train_dl:
                x,y = x.to(device), y.to(device)
                opt.zero_grad(); loss = crit(bb(x), y); loss.backward(); opt.step()
                total += loss.item(); n += 1
            scheduler.step()
            if ep % 10 == 0 or ep == 1:
                val_acc = test_accuracy(bb, val_dl, device)
                print(f' BB epoch {ep}/{epochs_bb} avg_loss {total/max(1,n):.4f} val_acc {val_acc:.4f}')
        bb_acc = test_accuracy(bb, test_dl, device)
        print(f'  Black-box model {name} test acc: {bb_acc:.4f}')
        save_checkpoint({'state': bb.state_dict()}, run_dir / f'blackbox_model_{name}.pt')

        # Dynamic distillation
        dyn_dir = run_dir / 'dynamic'; ensure_dir(dyn_dir)
        distiller = DynamicDistillerCIFAR(blackbox_model=bb, substitute_class=resnet20_cifar, device=device, image_nc=3, epsilon=8/255.0)
        # paper-faithful parameters
        dynamic_iters = 3
        adv_epochs = 30
        distill_epochs = 3
        static_epochs = 10
        logs, advgan = distiller.run(train_dl, iters=dynamic_iters, adv_epochs=adv_epochs, distill_epochs=distill_epochs, static_epochs=static_epochs, save_dir=dyn_dir)
        save_json(logs, dyn_dir / 'dynamic_logs.json')

        # Evaluate substitute and blackbox with final advgan
        gen_fn = lambda t: advgan.generate(t)
        sub_acc = test_accuracy(distiller.sub, test_dl, device)
        sub_asr_correct = evaluate_asr(distiller.sub, test_dl, gen_fn, device, only_on_correct=True)
        sub_asr_all = evaluate_asr(distiller.sub, test_dl, gen_fn, device, only_on_correct=False)
        bb_asr_correct = evaluate_asr(bb, test_dl, gen_fn, device, only_on_correct=True)
        bb_asr_all = evaluate_asr(bb, test_dl, gen_fn, device, only_on_correct=False)

        # Save artifacts (grids, CSVs, checkpoints)
        sub_dir = run_dir / 'substitute_eval'; ensure_dir(sub_dir)
        save_sample_grids_and_csv(distiller.sub, test_dl, gen_fn, sub_dir, device)
        bb_dir = run_dir / 'blackbox_eval'; ensure_dir(bb_dir)
        save_sample_grids_and_csv(bb, test_dl, gen_fn, bb_dir, device)

        save_checkpoint({'blackbox_model': bb.state_dict(), 'substitute_model': distiller.sub.state_dict(), 'advG': advgan.netG.state_dict(), 'advD': advgan.netD.state_dict()}, run_dir / 'final_models.pt')

        with open(run_dir / 'asr_summary.csv', 'w', newline='') as f:
            import csv
            w = csv.DictWriter(f, fieldnames=['model','accuracy','asr_only_correct','asr_all','eval_dir'])
            w.writeheader()
            w.writerow({'model':'substitute','accuracy':sub_acc,'asr_only_correct':sub_asr_correct,'asr_all':sub_asr_all,'eval_dir':str(sub_dir)})
            w.writerow({'model':f'blackbox_{name}','accuracy':bb_acc,'asr_only_correct':bb_asr_correct,'asr_all':bb_asr_all,'eval_dir':str(bb_dir)})

        final = {'blackbox':{'name':name,'accuracy':float(bb_acc)}, 'substitute':{'accuracy':float(sub_acc)}, 'ASR':{'substitute_only_correct':float(sub_asr_correct),'substitute_all':float(sub_asr_all),'blackbox_only_correct':float(bb_asr_correct),'blackbox_all':float(bb_asr_all)}, 'artifacts':{'dynamic_dir':str(dyn_dir),'substitute_eval_dir':str(sub_dir),'blackbox_eval_dir':str(bb_dir),'per_target_csv':str(run_dir / 'asr_summary.csv')}, 'logs':logs}
        save_json(final, run_dir / 'final_results.json')

        with open(master_csv, 'a', newline='') as f:
            import csv
            w = csv.DictWriter(f, fieldnames=['blackbox_model','blackbox_accuracy','substitute_accuracy','substitute_asr_only_correct','substitute_asr_all','blackbox_asr_only_correct','blackbox_asr_all','run_dir'])
            w.writerow({'blackbox_model':name,'blackbox_accuracy':bb_acc,'substitute_accuracy':sub_acc,'substitute_asr_only_correct':sub_asr_correct,'substitute_asr_all':sub_asr_all,'blackbox_asr_only_correct':bb_asr_correct,'blackbox_asr_all':bb_asr_all,'run_dir':str(run_dir)})

        print(f'Saved artifacts for {name} in {run_dir}')

    print('All targets complete. Master CSV:', master_csv)
    return str(root)

if __name__ == '__main__':
    main()
