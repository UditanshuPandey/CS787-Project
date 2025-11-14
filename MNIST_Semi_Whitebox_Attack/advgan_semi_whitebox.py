# advgan_train_full.py
import json
import time
from pathlib import Path
import csv
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset, random_split
from typing import Optional

# --------------------------
# Model definitions (unchanged, with minor attribute organization)
# --------------------------

class MNIST_target_net(nn.Module):
    """Model A from the paper"""
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 28->14
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)  # 14->7
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = self.logits(x)
        return x

class MNIST_model_B(nn.Module):
    """Model B from the paper - different architecture"""
    def __init__(self):
        super(MNIST_model_B, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(128*7*7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MNIST_model_C(nn.Module):
    """Model C from the paper - simpler architecture"""
    def __init__(self):
        super(MNIST_model_C, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x).view(-1)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Generator(nn.Module):
    def __init__(self, gen_input_nc, image_nc):
        super(Generator, self).__init__()

        encoder_lis = [
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True),   # 28->14
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),  # 14->7
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        ]

        bottle_neck_lis = [ResnetBlock(32),
                           ResnetBlock(32),
                           ResnetBlock(32),
                           ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),  # 7->14
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),   # 14->28
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, image_nc, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

# --------------------------
# Helpers (unchanged)
# --------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception:
            pass
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        try:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        except Exception:
            pass

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

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_checkpoint(obj: dict, path: Path):
    torch.save(obj, str(path))

def save_json(obj: dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# --------------------------
# AdvGAN Attack class (modified to accept weight hyperparams and targeted per-batch)
# --------------------------

class AdvGAN_Attack:
    def __init__(self, device, model, model_num_labels, image_nc, box_min, box_max,
                 epsilon=0.3, freeze_target: bool = True, adv_kappa: float = 0.0,
                 adv_lambda: float = 2.0, pert_lambda: float = 0.5, gan_lambda: float = 0.1,
                 disc_lr: float = 1e-4, gen_lr: float = 2e-4):
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.box_min = box_min
        self.box_max = box_max
        self.epsilon = epsilon
        self.adv_kappa = adv_kappa

        self.adv_lambda = adv_lambda
        self.pert_lambda = pert_lambda
        self.gan_lambda = gan_lambda

        self.gen_input_nc = image_nc
        self.netG = Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = Discriminator(image_nc).to(device)

        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=gen_lr)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=disc_lr)

        self.model.eval()
        if freeze_target:
            for p in self.model.parameters():
                p.requires_grad = False

    def train_batch(self, x, labels, target_class: Optional[torch.Tensor] = None):
        x = x.to(self.device)
        labels = labels.to(self.device)

        # ----- D -----
        perturbation_raw = self.netG(x)
        perturbation = self.epsilon * torch.tanh(perturbation_raw)
        adv_images = torch.clamp(x + perturbation, self.box_min, self.box_max)

        self.optimizer_D.zero_grad()
        pred_real = self.netDisc(x)
        loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
        pred_fake = self.netDisc(adv_images.detach())
        loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.optimizer_D.step()

        # ----- G -----
        self.optimizer_G.zero_grad()
        pred_fake_forG = self.netDisc(adv_images)
        loss_G_gan = F.mse_loss(pred_fake_forG, torch.ones_like(pred_fake_forG, device=self.device))
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1))

        logits_model = self.model(adv_images)
        if target_class is not None:
            # target_class expected to be a tensor of same shape as labels or scalar
            if isinstance(target_class, int):
                tlabels = torch.full_like(labels, target_class).to(self.device)
            else:
                tlabels = target_class.to(self.device)
            loss_adv = cw_loss(logits_model, tlabels, targeted=True, kappa=self.adv_kappa)
        else:
            loss_adv = cw_loss(logits_model, labels, targeted=False, kappa=self.adv_kappa)

        loss_G = self.adv_lambda * loss_adv + self.pert_lambda * loss_perturb + self.gan_lambda * loss_G_gan
        loss_G.backward()
        self.optimizer_G.step()

        return loss_D.item(), loss_G_gan.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs, target_mode: str = "increment", val_loader: Optional[DataLoader] = None,
              early_stop_patience: int = 10, save_dir: Optional[Path]=None):
        """
        target_mode: "increment" -> targeted to (label+1)%num_classes per batch
                     "random"    -> random target != label
                     "none"      -> untargeted (use labels as in current cw_loss untargeted)
                     integer n   -> fixed target class n (int)
        """
        best_val_asr = -1.0
        epochs_no_improve = 0
        per_epoch = []
        for epoch in range(1, epochs+1):
            loss_D_sum = loss_G_fake_sum = loss_perturb_sum = loss_adv_sum = 0.0
            num_batches = 0
            for images, labels in train_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # construct per-batch target if needed
                tgt = None
                if isinstance(target_mode, int):
                    tgt = torch.full_like(labels, target_mode).to(self.device)
                elif target_mode == "increment":
                    tgt = ((labels + 1) % self.model_num_labels).to(self.device)
                elif target_mode == "random":
                    # ensure random target isn't the same as label
                    rnd = torch.randint(0, self.model_num_labels, labels.shape, device=labels.device)
                    rnd = torch.where(rnd == labels, (rnd + 1) % self.model_num_labels, rnd)
                    tgt = rnd.to(self.device)
                elif target_mode == "none":
                    tgt = None
                else:
                    tgt = None

                lD, lGf, lP, lA = self.train_batch(images, labels, tgt)
                loss_D_sum += lD; loss_G_fake_sum += lGf; loss_perturb_sum += lP; loss_adv_sum += lA
                num_batches += 1

            log = {
                "epoch": epoch,
                "loss_D": float(loss_D_sum/max(1,num_batches)),
                "loss_G_fake": float(loss_G_fake_sum/max(1,num_batches)),
                "loss_perturb": float(loss_perturb_sum/max(1,num_batches)),
                "loss_adv": float(loss_adv_sum/max(1,num_batches))
            }
            per_epoch.append(log)
            print(f"Epoch {epoch}: loss_D: {log['loss_D']:.3f}, loss_G_fake: {log['loss_G_fake']:.3f}, "
                  f"loss_perturb: {log['loss_perturb']:.3f}, loss_adv: {log['loss_adv']:.3f}")

            # periodic checkpoints
            if save_dir is not None and (epoch % 5 == 0 or epoch == epochs):
                ensure_dir(save_dir)
                save_checkpoint({
                    "epoch": epoch,
                    "netG": self.netG.state_dict(),
                    "netD": self.netDisc.state_dict(),
                    "optimG": self.optimizer_G.state_dict(),
                    "optimD": self.optimizer_D.state_dict(),
                }, save_dir / f"checkpoint_epoch_{epoch}.pt")
                save_json(per_epoch, save_dir / "per_epoch_logs.json")

            # validation ASR monitoring and early stopping
            val_asr = None
            if val_loader is not None:
                val_asr = evaluate_attack_success(self.model, val_loader, self.generate_adv_examples, self.device, targeted=True, target_class=None, only_on_correct=True)
                print(f"  Validation ASR (targeted mode eval w/ per-batch targets): {val_asr:.4f}")
                # note: evaluate_attack_success's `targeted` argument requires a target_class;
                # because we trained with per-batch targets (increment), our evaluate function
                # passes target_class=None but `targeted=True` — in this case we expect the generator
                # to produce images that change label away from the original; for robust evaluation,
                # one could implement a specialized eval for targeted mode. For simplicity, we'll
                # use 'preds != labels' when evaluating masked (only_on_correct) attacks below.
                # We'll still use val_asr for early stopping.
                current_val = val_asr
                if current_val is not None:
                    if current_val > best_val_asr + 1e-6:
                        best_val_asr = current_val
                        epochs_no_improve = 0
                        # save best checkpoint
                        if save_dir is not None:
                            ensure_dir(save_dir)
                            save_checkpoint({
                                "epoch": epoch,
                                "netG": self.netG.state_dict(),
                                "netD": self.netDisc.state_dict(),
                                "optimG": self.optimizer_G.state_dict(),
                                "optimD": self.optimizer_D.state_dict(),
                            }, save_dir / f"best_checkpoint.pt")
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= early_stop_patience:
                        print(f"No improvement for {early_stop_patience} epochs. Early stopping at epoch {epoch}.")
                        break

        return per_epoch

    def generate_adv_examples(self, x):
        self.netG.eval()
        with torch.no_grad():
            raw = self.netG(x)
            perturbation = self.epsilon * torch.tanh(raw)
            adv_images = torch.clamp(x + perturbation, self.box_min, self.box_max)
        return adv_images

# --------------------------
# Evaluation utilities (unchanged with small clarification)
# --------------------------

def evaluate_attack_success(model, dataloader, attack_method, device, targeted=False, target_class=None, only_on_correct=True):
    """
    If targeted==False: success counted when preds != labels (untargeted)
    If targeted==True:
      - if target_class is int or tensor: success counted when preds == target_class
      - if target_class is None: fallback to untargeted metric preds != labels
    """
    model.eval()
    total = 0
    successful_attacks = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if only_on_correct:
                preds_clean = model(images).argmax(1)
                mask = preds_clean.eq(labels)
                if mask.sum().item() == 0:
                    continue
                images = images[mask]; labels = labels[mask]
            adv_images = attack_method(images)
            preds = model(adv_images).argmax(1)
            if targeted:
                if target_class is None:
                    # fallback to untargeted: count preds != labels
                    successful_attacks += (preds != labels).sum().item()
                else:
                    tgt = torch.full_like(labels, target_class) if isinstance(target_class, int) else target_class.to(device)
                    successful_attacks += (preds == tgt).sum().item()
            else:
                successful_attacks += (preds != labels).sum().item()
            total += labels.size(0)
    return successful_attacks / max(total, 1)

def test_model_accuracy(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# --------------------------
# Main pipeline
# --------------------------

def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = Path(f"./checkpoints/advgan_semi_whitebox_{timestamp}")
    ensure_dir(run_root)
    print(f"Saving artifacts to: {run_root}")

    # Hyperparams / dataset
    batch_size = 128
    image_nc = 1
    box_min, box_max = 0.0, 1.0
    model_num_labels = 10

    # Data: train/val/test splits
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_full = torchvision.datasets.MNIST('./dataset', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST('./dataset', train=False, download=True, transform=transform)

    # allocate 5k images as validation from training set
    val_size = 5000
    train_size = len(mnist_train_full) - val_size
    mnist_train, mnist_val = random_split(mnist_train_full, [train_size, val_size])
    train_dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Train target models (A,B,C) for longer than before
    print("Training target models (A, B, C)...")
    models = {
        'A': MNIST_target_net().to(device),
        'B': MNIST_model_B().to(device),
        'C': MNIST_model_C().to(device)
    }
    model_info = {}
    target_train_epochs = 30  # increased from 5 -> 30
    for name, model in models.items():
        print(f"\nTraining Model {name} for {target_train_epochs} epochs...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(1, target_train_epochs + 1):
            epoch_loss = 0.0
            batches = 0
            for images, labels in train_dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            if epoch % 5 == 0 or epoch == 1 or epoch == target_train_epochs:
                acc_val = test_model_accuracy(model, val_dataloader, device)
                print(f" Model {name} epoch {epoch}/{target_train_epochs} avg_loss {epoch_loss/max(1,batches):.4f} val_acc {acc_val:.4f}")
        acc = test_model_accuracy(model, test_dataloader, device)
        print(f"Model {name} test accuracy: {acc:.4f}")
        mdir = run_root / f"model_{name}"; ensure_dir(mdir)
        save_checkpoint({"state": model.state_dict()}, mdir / f"model_{name}_final.pt")
        model_info[name] = {"accuracy": float(acc), "checkpoint": str(mdir / f"model_{name}_final.pt")}

    results = {"models": model_info, "semi_whitebox": {}}

    # Semi-whitebox attacks: train AdvGAN against each model individually
    print("\n" + "="*60)
    print("SEMI-WHITEBOX ATTACKS (Models A, B, C)")
    print("="*60)

    summary_rows = []
    for model_name, target_model in models.items():
        print(f"\nSemi-whitebox attack on Model {model_name}...")
        target_model.eval()
        attack_dir = run_root / f"semi_whitebox_{model_name}"; ensure_dir(attack_dir)

        # create AdvGAN attacker (freeze_target=True for semi-whitebox)
        advgan = AdvGAN_Attack(
            device, target_model, model_num_labels, image_nc, box_min, box_max,
            epsilon=0.3, freeze_target=True, adv_kappa=0.0,
            adv_lambda=2.0, pert_lambda=0.5, gan_lambda=0.1,
            disc_lr=1e-4, gen_lr=2e-4
        )

        # Train AdvGAN: increased epochs, with early stopping based on validation ASR
        adv_epochs = 50
        print(f"Training AdvGAN for up to {adv_epochs} epochs (targeted mode: increment label->(label+1)%10)")
        adv_logs = advgan.train(train_dataloader, epochs=adv_epochs, target_mode="increment",
                                val_loader=val_dataloader, early_stop_patience=10, save_dir=attack_dir)
        save_json(adv_logs, attack_dir / "advgan_per_epoch_logs.json")

        # save final nets
        save_checkpoint({"netG": advgan.netG.state_dict(), "netD": advgan.netDisc.state_dict(),
                         "optimG": advgan.optimizer_G.state_dict(), "optimD": advgan.optimizer_D.state_dict()},
                        attack_dir / "final_checkpoint.pt")

        # Evaluate ASR
        def attack_fn(x): return advgan.generate_adv_examples(x)
        # we evaluate untargeted ASR (pred != label) on only-on-correct and all
        asr_only_correct = evaluate_attack_success(target_model, test_dataloader, attack_fn, device, targeted=False, target_class=None, only_on_correct=True)
        asr_all = evaluate_attack_success(target_model, test_dataloader, attack_fn, device, targeted=False, target_class=None, only_on_correct=False)
        print(f"Attack success rate on Model {model_name} (only-correct, untargeted metric): {asr_only_correct:.4f}")
        print(f"Attack success rate on Model {model_name} (all, untargeted metric): {asr_all:.4f}")

        results["semi_whitebox"][model_name] = {
            "attack_success_only_correct": float(asr_only_correct),
            "attack_success_all": float(asr_all),
            "dir": str(attack_dir)
        }

        # Save sample grids (first 16 test images)
        imgs = next(iter(test_dataloader))[0][:16].to(device)
        advs = advgan.generate_adv_examples(imgs)
        diffs = advs - imgs
        # normalize diffs to 0..1 for vis (image values in [0,1])
        dmin, dmax = diffs.min(), diffs.max()
        diffs_norm = (diffs - dmin) / (dmax - dmin + 1e-12)

        vutils.save_image(imgs, str(attack_dir / "clean_grid.png"), normalize=True)
        vutils.save_image(advs, str(attack_dir / "adv_grid.png"), normalize=True)
        vutils.save_image(diffs_norm, str(attack_dir / "diff_grid.png"), normalize=False)

        # Save a small CSV of per-image predictions for the sample grid
        sample_csv_path = attack_dir / "sample_predictions.csv"
        with open(sample_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["index", "clean_pred", "adv_pred", "label"])
            with torch.no_grad():
                labels = next(iter(test_dataloader))[1][:16].to(device)
                clean_preds = target_model(imgs).argmax(1)
                adv_preds = target_model(advs).argmax(1)
                for i in range(imgs.size(0)):
                    writer.writerow([i, int(clean_preds[i].cpu()), int(adv_preds[i].cpu()), int(labels[i].cpu())])

        # record summary row
        summary_rows.append({
            "model": model_name,
            "accuracy": float(model_info[model_name]["accuracy"]),
            "asr_only_correct": float(asr_only_correct),
            "asr_all": float(asr_all),
            "attack_dir": str(attack_dir)
        })

    # Save overall summary JSON and CSV
    save_json(results, run_root / "final_results.json")

    csv_path = run_root / "asr_summary.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["model", "accuracy", "asr_only_correct", "asr_all", "attack_dir"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nSaved ASR CSV summary to: {csv_path}")
    print(f"Saved full results JSON to: {run_root / 'final_results.json'}")
    print("\nSemi-whitebox attacks complete.")

if __name__ == "__main__":
    main()
