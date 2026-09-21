# loso_train.py
import os, time, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import EEGDataLoader, EEGDataset

from model import STGCN_PB_TFConv_T35_F5_FC



ROOT = r"E:\seedvig\data"
SUBJECTS = 23
BATCH = 256
EPOCHS = 150
LR = 5e-3
SEED = 42
WD = 2e-3
NUM_WORKERS = 0
PIN_MEMORY = True


RESULT_DIR = "result_stgcn_loso_persubject_norm"
CKPT_DIR = "checkpoints_stgcn_loso_persubject_norm"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


def set_seed(seed=42):
    ""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics_bin(y_true, y_pred):
    ""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def train_epoch(model, loader, loss_fn, opt, device):
    ""
    model.train()
    total_loss = 0.0
    all_true, all_pred = [], []
    for batch in loader:
        x, y = batch['input_e'].to(device), batch['target'].to(device).long()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * y.size(0)
        all_true.append(y.detach().cpu())
        all_pred.append(torch.argmax(logits, dim=1).detach().cpu())
    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    return total_loss / len(loader.dataset), metrics_bin(y_true, y_pred)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    ""
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []
    for batch in loader:
        x, y = batch['input_e'].to(device), batch['target'].to(device).long()
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)
        all_true.append(y.detach().cpu())
        all_pred.append(torch.argmax(logits, dim=1).detach().cpu())
    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    return total_loss / len(loader.dataset), metrics_bin(y_true, y_pred)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("Loading all data with per-subject normalization...")
    loader = EEGDataLoader(ROOT, subjects=SUBJECTS, min_samples_per_class=50)
    all_samples, subject_data, _, _ = loader.get_all()

    valid_subject_ids = sorted(list(subject_data.keys()))
    print(f"Data loaded. Starting with {len(valid_subject_ids)} eligible subjects for LOSO cross-validation.")

    all_fold_results = []


    A_hat, E, idx_flat, hw = loader.get_graph(device=device)


    for i, eval_subject_id in enumerate(valid_subject_ids):
        print("\n" + "=" * 80)
        print(f"### Fold {i + 1}/{len(valid_subject_ids)} | Held-out test subject: {eval_subject_id} ###")
        print("=" * 80)


        tr_idx, eval_idx = [], []
        for i_sample, sample in enumerate(all_samples):
            if sample['subject'] == eval_subject_id:
                eval_idx.append(i_sample)
            else:
                tr_idx.append(i_sample)

        print(f"Training samples: {len(tr_idx)}, test samples: {len(eval_idx)}")


        train_ds = EEGDataset([all_samples[i] for i in tr_idx], subject_data, idx_flat, hw)
        eval_ds = EEGDataset([all_samples[i] for i in eval_idx], subject_data, idx_flat, hw)

        trn = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        evl = DataLoader(eval_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


        set_seed(SEED)
        model = STGCN_PB_TFConv_T35_F5_FC(A_hat=A_hat, Freq=5, num_classes=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.6, patience=10, min_lr=1e-6)
        criterion = nn.CrossEntropyLoss()


        best_eval_acc, best_ep, best_eval_metrics = -1.0, -1, None
        fold_hist = []
        for ep in range(1, EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_met = train_epoch(model, trn, criterion, opt, device)
            eval_loss, eval_met = eval_epoch(model, evl, criterion, device)
            t1 = time.time()

            if eval_met['acc'] > best_eval_acc:
                best_eval_acc = eval_met['acc']
                best_ep = ep
                best_eval_metrics = eval_met
                best_eval_metrics['loss'] = eval_loss
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"best_model_for_{eval_subject_id}.pth"))

            sched.step(eval_loss)

            print(f"Epoch {ep}/{EPOCHS} | Time: {t1 - t0:.1f}s | Train Loss: {tr_loss:.4f}, Acc: {tr_met['acc']:.4f} | "
                  f"Eval Loss: {eval_loss:.4f}, Acc: {eval_met['acc']:.4f}")

            fold_hist.append({'epoch': ep, 'train_loss': tr_loss, 'train_acc': tr_met['acc'], 'eval_loss': eval_loss,
                              'eval_acc': eval_met['acc']})

        pd.DataFrame(fold_hist).to_csv(os.path.join(RESULT_DIR, f"history_{eval_subject_id}.csv"), index=False)
        print(f"\nFold complete. Best ACC for subject {eval_subject_id}: {best_eval_acc:.4f} (at epoch {best_ep})")


        fold_result = {
            'eval_subject': eval_subject_id,
            'best_epoch': best_ep,
            'test_loss': best_eval_metrics['loss'],
            'test_acc': best_eval_metrics['acc'],
            'test_precision': best_eval_metrics['precision'],
            'test_recall': best_eval_metrics['recall'],
            'test_f1': best_eval_metrics['f1'],
        }
        all_fold_results.append(fold_result)
        pd.DataFrame([fold_result]).to_csv(os.path.join(RESULT_DIR, f"result_{eval_subject_id}.csv"), index=False)


    print("\n\n" + "=" * 80)
    print("### Final LOSO cross-validation results (per-subject normalization) ###")
    print("=" * 80)

    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(os.path.join(RESULT_DIR, "summary_all_folds.csv"), index=False)

    mean_results = results_df.mean(numeric_only=True)
    std_results = results_df.std(numeric_only=True)

    print("--- Results by fold ---")
    print(results_df.to_string())
    print("\n--- Summary results (mean ± standard deviation) ---")
    print(f"Mean accuracy (Accuracy):    {mean_results['test_acc']:.4f} ± {std_results['test_acc']:.4f}")
    print(f"Mean precision (Precision):   {mean_results['test_precision']:.4f} ± {std_results['test_precision']:.4f}")
    print(f"Mean recall (Recall):      {mean_results['test_recall']:.4f} ± {std_results['test_recall']:.4f}")
    print(f"Mean F1-score (F1-Score):    {mean_results['test_f1']:.4f} ± {std_results['test_f1']:.4f}")

    summary_stats = {
        'mean_acc': mean_results['test_acc'], 'std_acc': std_results['test_acc'],
        'mean_precision': mean_results['test_precision'], 'std_precision': std_results['test_precision'],
        'mean_recall': mean_results['test_recall'], 'std_recall': std_results['test_recall'],
        'mean_f1': mean_results['test_f1'], 'std_f1': std_results['test_f1'],
    }
    pd.DataFrame([summary_stats]).to_csv(os.path.join(RESULT_DIR, "summary_aggregated.csv"), index=False)


if __name__ == "__main__":
    main()
