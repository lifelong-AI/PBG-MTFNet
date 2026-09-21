# loso_train.py
import os, time, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
# 导入您提供的、执行被试者内归一化的数据加载器
from data_loader import EEGDataLoader, EEGDataset
# 导入您的模型
from model import STGCN_PB_TFConv_T35_F5_FC

# ===== 配置 =====
# 请根据您的实际情况修改数据根目录
ROOT = r"E:\seedvig\data"
SUBJECTS = 23
BATCH = 256
EPOCHS = 150
LR = 5e-3
SEED = 42
WD = 2e-3
NUM_WORKERS = 0
PIN_MEMORY = True

# 为LOSO实验创建独立的结果和模型保存目录
RESULT_DIR = "result_stgcn_loso_persubject_norm"
CKPT_DIR = "checkpoints_stgcn_loso_persubject_norm"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


def set_seed(seed=42):
    """设置随机种子以保证实验可复现。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics_bin(y_true, y_pred):
    """计算二分类任务的常用评估指标。"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def train_epoch(model, loader, loss_fn, opt, device):
    """模型单轮训练函数。"""
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
    """模型单轮评估函数。"""
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

    # 1) 一次性加载所有数据（归一化已在内部完成）
    print("正在加载并按被试者归一化所有数据...")
    loader = EEGDataLoader(ROOT, subjects=SUBJECTS, min_samples_per_class=50)
    all_samples, subject_data, _, _ = loader.get_all()

    valid_subject_ids = sorted(list(subject_data.keys()))
    print(f"数据加载完成。开始对 {len(valid_subject_ids)} 位合格被试进行LOSO交叉验证。")

    all_fold_results = []

    # 获取图邻接矩阵（所有折都一样，只需获取一次）
    A_hat, E, idx_flat, hw = loader.get_graph(device=device)

    # 2) 开始LOSO主循环
    for i, eval_subject_id in enumerate(valid_subject_ids):
        print("\n" + "=" * 80)
        print(f"### 第 {i + 1}/{len(valid_subject_ids)} 折 | 留出测试被试: {eval_subject_id} ###")
        print("=" * 80)

        # 3) 划分当前折的训练集和测试集索引
        tr_idx, eval_idx = [], []
        for i_sample, sample in enumerate(all_samples):
            if sample['subject'] == eval_subject_id:
                eval_idx.append(i_sample)
            else:
                tr_idx.append(i_sample)

        print(f"训练样本数: {len(tr_idx)}, 测试样本数: {len(eval_idx)}")

        # 4) 创建数据集和加载器（注意：EEGDataset不再需要mean/std参数）
        train_ds = EEGDataset([all_samples[i] for i in tr_idx], subject_data, idx_flat, hw)
        eval_ds = EEGDataset([all_samples[i] for i in eval_idx], subject_data, idx_flat, hw)

        trn = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        evl = DataLoader(eval_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        # 5) 为每一折重新初始化模型、优化器等
        set_seed(SEED)
        model = STGCN_PB_TFConv_T35_F5_FC(A_hat=A_hat, Freq=5, num_classes=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.6, patience=10, min_lr=1e-6)
        criterion = nn.CrossEntropyLoss()

        # 6) 针对当前折进行训练和评估
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
        print(f"\n当前折完成。在被试 {eval_subject_id} 上的最佳ACC为: {best_eval_acc:.4f} (出现在第 {best_ep} 个epoch)")

        # 7) 记录本轮的最佳结果
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

    # 8) 汇总并报告所有折的最终结果
    print("\n\n" + "=" * 80)
    print("### LOSO 交叉验证最终结果汇总 (按被试者归一化) ###")
    print("=" * 80)

    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(os.path.join(RESULT_DIR, "summary_all_folds.csv"), index=False)

    mean_results = results_df.mean(numeric_only=True)
    std_results = results_df.std(numeric_only=True)

    print("--- 各折独立结果 ---")
    print(results_df.to_string())
    print("\n--- 汇总结果 (均值 ± 标准差) ---")
    print(f"平均准确率 (Accuracy):    {mean_results['test_acc']:.4f} ± {std_results['test_acc']:.4f}")
    print(f"平均精确率 (Precision):   {mean_results['test_precision']:.4f} ± {std_results['test_precision']:.4f}")
    print(f"平均召回率 (Recall):      {mean_results['test_recall']:.4f} ± {std_results['test_recall']:.4f}")
    print(f"平均F1分数 (F1-Score):    {mean_results['test_f1']:.4f} ± {std_results['test_f1']:.4f}")

    summary_stats = {
        'mean_acc': mean_results['test_acc'], 'std_acc': std_results['test_acc'],
        'mean_precision': mean_results['test_precision'], 'std_precision': std_results['test_precision'],
        'mean_recall': mean_results['test_recall'], 'std_recall': std_results['test_recall'],
        'mean_f1': mean_results['test_f1'], 'std_f1': std_results['test_f1'],
    }
    pd.DataFrame([summary_stats]).to_csv(os.path.join(RESULT_DIR, "summary_aggregated.csv"), index=False)


if __name__ == "__main__":
    main()