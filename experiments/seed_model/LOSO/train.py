# train.py
import os, time, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import EEGDataLoader, EEGDataset, compute_stats_crossband
from model import STGCN_PB_TFConv_T35_F5_FC

ROOT        = r"/tmp/mycode/data"
SUBJECTS    = 23
BATCH       = 256
EPOCHS      = 150
LR          = 5e-3
SEED        = 42
WD          = 6e-2
SPLITS      = (0.6, 0.2, 0.2)
NUM_WORKERS = 0
PIN_MEMORY  = True


RESULT_DIR = "result_stgcn_strict"
CKPT_DIR   = "checkpoints_stgcn_strict"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def split_indices(n_total, splits=(0.6,0.2,0.2), seed=42):
    assert abs(sum(splits)-1.0) < 1e-6
    rng = np.random.RandomState(seed); idx = np.arange(n_total); rng.shuffle(idx)
    n_tr = int(n_total*splits[0]); n_va = int(n_total*splits[1])
    return idx[:n_tr].tolist(), idx[n_tr:n_tr+n_va].tolist(), idx[n_tr+n_va:].tolist()

def metrics_bin(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

def train_epoch(model, loader, loss_fn, opt, device):
    model.train(); tot=0.0; ys=[]; ps=[]
    for batch in loader:
        x = batch['input_e'].to(device)   # [B,T,F,E]
        y = batch['target'].to(device).long()
        opt.zero_grad(set_to_none=True)
        logits = model(x)                 # [B,2]
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()*y.size(0)
        preds = torch.argmax(logits, dim=1)
        ys.append(y.detach().cpu()); ps.append(preds.detach().cpu())
    y_true = torch.cat(ys).numpy(); y_pred = torch.cat(ps).numpy()
    return tot/len(loader.dataset), metrics_bin(y_true, y_pred)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval(); tot=0.0; ys=[]; ps=[]
    for batch in loader:
        x = batch['input_e'].to(device)   # [B,T,F,E]
        y = batch['target'].to(device).long()
        logits = model(x)
        loss = loss_fn(logits, y)
        tot += loss.item()*y.size(0)
        preds = torch.argmax(logits, dim=1)
        ys.append(y.detach().cpu()); ps.append(preds.detach().cpu())
    y_true = torch.cat(ys).numpy(); y_pred = torch.cat(ps).numpy()
    return tot/len(loader.dataset), metrics_bin(y_true, y_pred)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    loader = EEGDataLoader(ROOT, subjects=SUBJECTS)
    all_samples, subject_data, space_mask_hw, _ = loader.get_all()
    n_total = len(all_samples)
    tr_idx, va_idx, te_idx = split_indices(n_total, SPLITS, SEED)

    mean_hw, std_hw = compute_stats_crossband(tr_idx, all_samples, subject_data, space_mask_hw)


    A_hat, E, idx_flat, hw = loader.get_graph(device=device)


    train_ds = EEGDataset([all_samples[i] for i in tr_idx], subject_data, mean_hw, std_hw, idx_flat, hw)
    val_ds   = EEGDataset([all_samples[i] for i in va_idx], subject_data, mean_hw, std_hw, idx_flat, hw)
    test_ds  = EEGDataset([all_samples[i] for i in te_idx], subject_data, mean_hw, std_hw, idx_flat, hw)

    trn = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    tst = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


    model = STGCN_PB_TFConv_T35_F5_FC(A_hat=A_hat, Freq=5, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.6, patience=10, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()


    best_acc, best_ep, best_val = -1.0, -1, None
    hist = []
    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_met = train_epoch(model, trn, criterion, opt, device)
        va_loss, va_met = eval_epoch(model, val, criterion, device)


        if va_met['acc'] > best_acc:
            best_acc, best_ep, best_val = va_met['acc'], ep, va_met
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_acc.pth"))
            print(f">>> Best (ACC) model updated: ACC={best_acc:.6f} (epoch {best_ep})")


        sched.step(va_loss)

        print(f"Epoch {ep}/{EPOCHS} | Time: {time.time()-t0:.0f}s")
        print(f"Train | Loss: {tr_loss:.4f} | Acc: {tr_met['acc']:.4f} | Prec: {tr_met['precision']:.4f} | Rec: {tr_met['recall']:.4f} | F1: {tr_met['f1']:.4f}")
        print(f"Val   | Loss: {va_loss:.4f} | Acc: {va_met['acc']:.4f} | Prec: {va_met['precision']:.4f} | Rec: {va_met['recall']:.4f} | F1: {va_met['f1']:.4f}\n")

        hist.append({
            'epoch': ep,
            'train_loss': tr_loss, 'train_acc': tr_met['acc'], 'train_precision': tr_met['precision'],
            'train_recall': tr_met['recall'], 'train_f1': tr_met['f1'],
            'val_loss': va_loss,   'val_acc': va_met['acc'],   'val_precision': va_met['precision'],
            'val_recall': va_met['recall'], 'val_f1': va_met['f1'],
        })

    pd.DataFrame(hist).to_csv(os.path.join(RESULT_DIR, "history.csv"), index=False)


    ckpt = os.path.join(CKPT_DIR, "best_acc.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    te_loss, te_met = eval_epoch(model, tst, criterion, device)
    print("=== Test (best on Val-ACC) ===")
    print(f"Loss: {te_loss:.4f} | Acc: {te_met['acc']:.4f} | Prec: {te_met['precision']:.4f} | Rec: {te_met['recall']:.4f} | F1: {te_met['f1']:.4f}")

    pd.DataFrame([{
        'best_epoch': best_ep,
        'val_best_acc': best_acc,
        'test_loss': te_loss, 'test_acc': te_met['acc'],
        'test_precision': te_met['precision'], 'test_recall': te_met['recall'], 'test_f1': te_met['f1'],
    }]).to_csv(os.path.join(RESULT_DIR, "test_acc_selected.csv"), index=False)

if __name__ == "__main__":
    main()
