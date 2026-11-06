from collections import Counter
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from seqeval.metrics import classification_report as entity_classification_report
from seqeval.scheme import IOB2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn

from data import PAD, PAD_TAG, SENTIMENT_LABELS, encode_tokens, tokenize_vsfc
from engine import evaluate_cls, evaluate_ner
from models import n_params, n_trainable_params


def set_seed(seed: int = 42) -> None:
    #Co dinh seed toan cuc de dam bao tinh tai lap giua cac bai
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_dataset_overview(
    vsfc: dict[str, list[dict]],
    ner: dict[str, list[dict]],
    labels: list[str] = SENTIMENT_LABELS,
    save_path: str | Path | None = None,
) -> None:
    #Truc quan hoa phan bo nhan UIT-VSFC (mat can bang lop neutral) va phan bo do dai cau
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    x = np.arange(len(labels))
    width = 0.26
    colors = {"train": "#2b6cb0", "dev": "#38a169", "test": "#ed8936"}
    for idx, split in enumerate(("train", "dev", "test")):
        cnt = Counter(r["sentiment"] for r in vsfc[split])
        vals = [cnt[lb] for lb in labels]
        bars = ax1.bar(
            x + (idx - 1) * width,
            vals,
            width,
            label=f"{split} ({len(vsfc[split]):,})",
            color=colors[split],
        )
        for b in bars:
            h = int(b.get_height())
            ax1.text(b.get_x() + b.get_width() / 2, h + 60, f"{h:,}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x, labels, fontsize=10)
    ax1.set_ylabel("So luong cau")
    ax1.set_title("Phan bo 3 nhan cam xuc tren UIT-VSFC", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    vsfc_lens = [len(tokenize_vsfc(r["sentence"])) for r in vsfc["train"]]
    ner_lens = [len(r["words"]) for r in ner["train"]]
    ax2.hist(vsfc_lens, bins=40, alpha=0.65, label=f"UIT-VSFC train (median={int(np.median(vsfc_lens))}, max={max(vsfc_lens)})", color="#3182ce")
    ax2.hist(ner_lens, bins=40, alpha=0.65, label=f"PhoNER train (median={int(np.median(ner_lens))}, max={max(ner_lens)})", color="#dd6b20")
    ax2.set_xlabel("So token / syllable trong cau")
    ax2.set_ylabel("So cau")
    ax2.set_title("Phan bo do dai chuoi (train) -> Ly do can Packing", fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


@torch.no_grad()
def verify_cls_packing(model: nn.Module, vocab: dict[str, int], device) -> float:
    #Kiem tra cung 1 cau khi dem den 8 va 64 vi tri: co packing phai cho cung logits
    sent_ids = encode_tokens(tokenize_vsfc("giảng viên dạy hay"), vocab)
    lens = torch.tensor([len(sent_ids)], dtype=torch.long)

    def _pad_to(total_len: int) -> torch.Tensor:
        return torch.tensor([sent_ids + [PAD] * (total_len - len(sent_ids))], dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    logits_8 = model(_pad_to(8), lens)
    logits_64 = model(_pad_to(64), lens)
    torch.testing.assert_close(logits_8, logits_64, atol=1e-6, rtol=1e-5)

    raw_8, _ = model.rnn(model.emb(_pad_to(8)))
    raw_64, _ = model.rnn(model.emb(_pad_to(64)))
    unpacked_diff = float((raw_8[:, -1] - raw_64[:, -1]).abs().max().item())
    if was_training:
        model.train()
    return unpacked_diff


@torch.no_grad()
def verify_ner_packing_and_loss(
    model: nn.Module,
    ner_vocab: dict[str, int],
    tag_to_id: dict[str, int],
    num_tags: int,
    device,
    pad_tag: int = PAD_TAG,
) -> None:
    #Kiem tra BiLSTM dem 8 vs 64 va CrossEntropyLoss(ignore_index=-100) tren vi du tu tao
    toy_ids = encode_tokens(["Ví", "dụ", "tự", "tạo"], ner_vocab)

    def _pad_ner(total_len: int) -> torch.Tensor:
        return torch.tensor([toy_ids + [PAD] * (total_len - len(toy_ids))], dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    out_8 = model(_pad_ner(8), torch.tensor([4], dtype=torch.long))
    out_64 = model(_pad_ner(64), torch.tensor([4], dtype=torch.long))
    torch.testing.assert_close(out_8[:, :4], out_64[:, :4], atol=1e-6, rtol=1e-5)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_tag)
    target_4 = torch.tensor([[tag_to_id["O"]] * 4], dtype=torch.long, device=device)
    loss_clean = criterion(out_8[:, :4].reshape(-1, num_tags), target_4.flatten())

    target_8 = torch.cat([target_4, torch.full((1, 4), pad_tag, dtype=torch.long, device=device)], dim=1)
    loss_padded = criterion(out_8.reshape(-1, num_tags), target_8.flatten())
    torch.testing.assert_close(loss_clean, loss_padded, atol=1e-6, rtol=1e-5)

    if was_training:
        model.train()


def plot_cls_history(
    history: list[dict],
    title: str = "",
    save_path: str | Path | None = None,
) -> None:
    #Ve duong cong train/dev loss va accuracy + dev macro-F1 theo epoch
    if not history:
        return

    ep = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))

    ax1.plot(ep, [h["train_loss"] for h in history], "b-o", markersize=4, label="train loss")
    ax1.plot(ep, [h["dev_loss"] for h in history], "r--s", markersize=4, label="dev loss")
    ax1.set_title(f"{title} - Loss" if title else "Loss theo epoch", fontweight="bold")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("CrossEntropy loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(ep, [h["train_acc"] for h in history], "b-o", markersize=4, label="train acc")
    ax2.plot(ep, [h["dev_acc"] for h in history], "g--^", markersize=4, label="dev acc")
    ax2.plot(ep, [h["dev_f1"] for h in history], "m-s", markersize=4, label="dev macro-F1")
    ax2.set_title(f"{title} - Accuracy & macro-F1" if title else "Accuracy & macro-F1", fontweight="bold")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_cls_multi_curves(
    named_histories: list[tuple[str, list[dict]]],
    title: str = "So sanh qua trinh hoi tu",
    save_path: str | Path | None = None,
) -> None:
    #Dat canh nhau duong train loss, dev loss va dev macro-F1 giua cac kien truc (LSTM vs GRU, hoac do sau 1/2/5)
    valid = [(n, h) for n, h in named_histories if h]
    if not valid:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    markers = ["o", "s", "^", "D"]
    for idx, (name, hist) in enumerate(valid):
        ep = [r["epoch"] for r in hist]
        mk = markers[idx % len(markers)]
        axes[0].plot(ep, [r["train_loss"] for r in hist], marker=mk, markersize=4, label=name)
        axes[1].plot(ep, [r["dev_loss"] for r in hist], marker=mk, markersize=4, linestyle="--", label=name)
        axes[2].plot(ep, [r["dev_f1"] for r in hist], marker=mk, markersize=4, label=name)

    axes[0].set_title(f"{title} - Train Loss", fontweight="bold")
    axes[1].set_title(f"{title} - Dev Loss", fontweight="bold")
    axes[2].set_title(f"{title} - Dev macro-F1", fontweight="bold")
    axes[0].set_ylabel("CrossEntropy loss")
    axes[2].set_ylabel("macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> None:
    #Ve va luu ma tran nham lan cho 3 lop sentiment UIT-VSFC
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, fontsize=9.5)
    ax.set_yticks(range(len(classes)), classes, fontsize=9.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(
                j,
                i,
                f"{val:,}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def evaluate_tfidf_test(
    pipeline,
    vsfc_test: list[dict],
    label_to_id: dict[str, int],
    labels: list[str],
    summary: dict,
    out_dir: str | Path | None = None,
) -> dict:
    #Danh gia baseline TF-IDF + LogisticRegression tren tap test UIT-VSFC
    test_text = [r["sentence"] for r in vsfc_test]
    y_true = [label_to_id[r["sentiment"]] for r in vsfc_test]
    y_pred = pipeline.predict(test_text).tolist()

    label_ids = list(range(len(labels)))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    print("KET QUA BASELINE TF-IDF + LOGISTIC REGRESSION (UIT-VSFC TEST):")
    print(f"    - Cau hinh          : {summary['config_str']}")
    print(f"    - Thoi gian fit     : {summary['fit_sec']:.2f}s (Majority dev macro-F1 = {summary['majority_dev_f1']:.4f})")
    print(f"    - Dev macro-F1      : {summary['best_dev_f1']:.4f}")
    print(f"    - Test Accuracy     : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"    - Precision (macro) : {prec:.4f}")
    print(f"    - Recall (macro)    : {rec:.4f}")
    print(f"    - macro-F1 (chinh)  : {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, labels=label_ids, target_names=labels, digits=4, zero_division=0))

    if out_dir is not None:
        plot_confusion_matrix(
            cm,
            labels,
            title="Confusion Matrix - TF-IDF + LogReg (Test)",
            save_path=Path(out_dir) / "confusion_matrix_tfidf.png",
        )

    return {
        "model": "TF-IDF + Logistic Regression",
        "task": "Sentiment (macro-F1)",
        "params": summary["params"],
        "params_note": summary["config_str"],
        "best_epoch": None,
        "best_dev_f1": summary["best_dev_f1"],
        "sec_per_epoch": summary["fit_sec"],
        "sec_note": f"{summary['fit_sec']:.2f}s (fit)",
        "test_loss": None,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "test_f1": macro_f1,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(labels, per_class_f1.tolist())),
        "cm": cm,
    }


def evaluate_cls_test(
    model: nn.Module,
    test_dl,
    labels: list[str],
    device,
    model_name: str,
    summary: dict,
    out_dir: str | Path | None = None,
) -> dict:
    #Danh gia day du mo hinh hoi quy tren tap test UIT-VSFC va ve Confusion Matrix 3x3
    loss, acc, macro_f1, y_true, y_pred = evaluate_cls(
        model, test_dl, nn.CrossEntropyLoss(), device, num_classes=len(labels)
    )
    assert not model.training, "model.training phai la False trong khi danh gia!"

    label_ids = list(range(len(labels)))
    prec = precision_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    total_p = n_params(model)
    train_p = n_trainable_params(model)
    sec_ep = summary.get("sec_per_epoch")
    best_ep = summary.get("best_epoch")
    best_dev = summary.get("best_dev_f1")

    print(f"KET QUA TAP TEST - {model_name}:")
    print(f"    - Tong so tham so   : {total_p:,} (cap nhat: {train_p:,})")
    if best_ep is not None and best_dev is not None:
        print(f"    - Checkpoint dev    : epoch {best_ep} (dev macro-F1 = {best_dev:.4f})")
    if sec_ep is not None:
        print(f"    - Thoi gian / epoch : {sec_ep:.2f}s")
    print(f"    - Test Loss         : {loss:.4f}")
    print(f"    - Accuracy          : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"    - Precision (macro) : {prec:.4f}")
    print(f"    - Recall (macro)    : {rec:.4f}")
    print(f"    - macro-F1 (chinh)  : {macro_f1:.4f}\n")

    print("Bang Precision, Recall, F1 tung lop tren tap Test UIT-VSFC:")
    print(classification_report(y_true, y_pred, labels=label_ids, target_names=labels, digits=4, zero_division=0))

    cm_path = (Path(out_dir) / "confusion_matrix.png") if out_dir is not None else None
    plot_confusion_matrix(cm, labels, title=f"Confusion Matrix - {model_name} (Test)", save_path=cm_path)

    return {
        "model": model_name,
        "task": "Sentiment (macro-F1)",
        "params": total_p,
        "trainable_params": train_p,
        "best_epoch": best_ep,
        "best_dev_f1": best_dev,
        "sec_per_epoch": sec_ep,
        "sec_note": f"{sec_ep:.2f}s" if sec_ep is not None else "—",
        "test_loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "test_f1": macro_f1,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(labels, per_class_f1.tolist())),
        "cm": cm,
    }


@torch.no_grad()
def analyze_cls_dev_errors(
    model: nn.Module,
    dev_dl,
    vsfc_dev: list[dict],
    labels: list[str],
    device,
    max_cases: int = 5,
) -> list[dict]:
    #Phan tich it nhat 3 truong hop du doan sai tieu bieu tren tap dev UIT-VSFC (khong dung test)
    model.eval()
    probs_all, preds_all, trues_all = [], [], []
    for ids, lengths, y in dev_dl:
        logits = model(ids.to(device), lengths)
        probs = torch.softmax(logits, dim=-1).cpu()
        probs_all.extend(probs.tolist())
        preds_all.extend(probs.argmax(dim=-1).tolist())
        trues_all.extend(y.tolist())

    pair_counts = Counter()
    errors = []
    for idx, (row, y_t, y_p, pr) in enumerate(zip(vsfc_dev, trues_all, preds_all, probs_all)):
        if y_t != y_p:
            pair_counts[(labels[y_t], labels[y_p])] += 1
            errors.append(
                {
                    "dev_idx": idx,
                    "sentence": row["sentence"],
                    "true": labels[y_t],
                    "pred": labels[y_p],
                    "conf": float(pr[y_p]),
                    "true_prob": float(pr[y_t]),
                }
            )

    print(f"Tong so cau du doan sai tren dev: {len(errors)}/{len(vsfc_dev)} ({len(errors)/len(vsfc_dev)*100:.2f}%)")
    print("Phan bo cac cap nhan bi nham tren dev (Nhan that -> Du doan):")
    for (t_lb, p_lb), cnt in pair_counts.most_common():
        print(f"    - {t_lb:<10} -> {p_lb:<10} : {cnt:>3} cau")

    #Chon da dang cac kieu nham tieu bieu (neutral bi nham, cau trai chieu negative <-> positive, ...)
    selected = []
    seen_pairs = set()
    for err in sorted(errors, key=lambda e: e["conf"], reverse=True):
        pair = (err["true"], err["pred"])
        if pair not in seen_pairs:
            selected.append(err)
            seen_pairs.add(pair)
        if len(selected) >= max_cases:
            break
    if len(selected) < max_cases:
        used_ids = {e["dev_idx"] for e in selected}
        for err in errors:
            if err["dev_idx"] not in used_ids:
                selected.append(err)
            if len(selected) >= max_cases:
                break

    print(f"\nChi tiet {len(selected)} truong hop du doan sai tieu bieu tren tap dev:")
    for rank, e in enumerate(selected, start=1):
        print(
            f"  [{rank}] dev#{e['dev_idx']} | That: {e['true']} (p={e['true_prob']:.3f}) "
            f"-> Du doan: {e['pred']} (conf={e['conf']:.3f})\n"
            f"      Cau: \"{e['sentence']}\""
        )
    print()
    return selected


def plot_ner_history(
    history: list[dict],
    title: str = "BiLSTM 5 lop - PhoNER",
    save_path: str | Path | None = None,
) -> None:
    #Ve duong cong token loss va entity micro-F1 (strict IOB2) tren PhoNER
    if not history:
        return

    ep = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))

    ax1.plot(ep, [h["train_loss"] for h in history], "b-o", markersize=4, label="train token loss")
    ax1.plot(ep, [h["dev_loss"] for h in history], "r--s", markersize=4, label="dev token loss")
    ax1.set_title(f"{title} - Token CrossEntropy Loss", fontweight="bold")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (ignore PAD_TAG=-100)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(ep, [h["dev_prec"] for h in history], "g--^", markersize=4, label="dev Precision")
    ax2.plot(ep, [h["dev_rec"] for h in history], "c--v", markersize=4, label="dev Recall")
    ax2.plot(ep, [h["dev_f1"] for h in history], "m-s", markersize=4, label="dev Entity micro-F1")
    ax2.set_title(f"{title} - Strict IOB2 Entity Metrics", fontweight="bold")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def evaluate_ner_test(
    model: nn.Module,
    test_dl,
    tags: list[str],
    device,
    model_name: str,
    summary: dict,
    out_dir: str | Path | None = None,
    pad_tag: int = PAD_TAG,
) -> dict:
    #Danh gia BiLSTM NER tren tap test PhoNER bang seqeval strict IOB2 va ve bieu do F1 10 loai thuc the
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tag)
    loss, ent_prec, ent_rec, ent_f1, true_seqs, pred_seqs = evaluate_ner(
        model, test_dl, criterion, tags, device, pad_tag=pad_tag
    )
    assert not model.training, "model.training phai la False trong khi danh gia!"

    total_p = n_params(model)
    train_p = n_trainable_params(model)
    sec_ep = summary.get("sec_per_epoch")
    best_ep = summary.get("best_epoch")
    best_dev = summary.get("best_dev_f1")

    report_str = entity_classification_report(
        true_seqs, pred_seqs, mode="strict", scheme=IOB2, digits=4, zero_division=0
    )
    report_dict = entity_classification_report(
        true_seqs, pred_seqs, mode="strict", scheme=IOB2, digits=4, zero_division=0, output_dict=True
    )

    print(f"KET QUA TAP TEST - {model_name} (PhoNER syllable):")
    print(f"    - Tong so tham so       : {total_p:,} (cap nhat: {train_p:,})")
    if best_ep is not None and best_dev is not None:
        print(f"    - Checkpoint dev        : epoch {best_ep} (dev entity-F1 = {best_dev:.4f})")
    if sec_ep is not None:
        print(f"    - Thoi gian / epoch     : {sec_ep:.2f}s")
    print(f"    - Test Token Loss       : {loss:.4f}")
    print(f"    - Entity Precision      : {ent_prec:.4f}")
    print(f"    - Entity Recall         : {ent_rec:.4f}")
    print(f"    - Entity micro-F1 (IOB2): {ent_f1:.4f}\n")
    print("Bang Precision, Recall, F1 tung loai thuc the (seqeval strict IOB2):")
    print(report_str)

    entity_f1_map = {
        k: float(v["f1-score"])
        for k, v in report_dict.items()
        if isinstance(v, dict) and k not in ("micro avg", "macro avg", "weighted avg")
    }

    if out_dir is not None:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "ner_test_report.txt").write_text(report_str, encoding="utf-8")

    if entity_f1_map:
        sorted_items = sorted(entity_f1_map.items(), key=lambda kv: kv[1])
        ent_names = [k for k, _ in sorted_items]
        ent_scores = [v for _, v in sorted_items]

        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        bars = ax.barh(ent_names, ent_scores, color="#3182ce")
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("Strict IOB2 F1-score")
        ax.set_title(f"F1 tung loai thuc the tren tap Test PhoNER - {model_name}", fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for b in bars:
            w = b.get_width()
            ax.text(w + 0.012, b.get_y() + b.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.5)
        fig.tight_layout()
        if out_dir is not None:
            fig.savefig(Path(out_dir) / "entity_f1_bars.png", dpi=120, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    return {
        "model": model_name,
        "task": "NER (Entity micro-F1)",
        "params": total_p,
        "trainable_params": train_p,
        "best_epoch": best_ep,
        "best_dev_f1": best_dev,
        "sec_per_epoch": sec_ep,
        "sec_note": f"{sec_ep:.2f}s" if sec_ep is not None else "—",
        "test_loss": loss,
        "precision": ent_prec,
        "recall": ent_rec,
        "test_f1": ent_f1,
        "entity_f1": ent_f1,
        "per_entity_f1": entity_f1_map,
    }


def extract_iob2_spans(seq: list[str]) -> list[tuple[int, int, str]]:
    #Trich xuat danh sach thuc the (start, end_inclusive, entity_type) theo quy uoc IOB2
    spans = []
    cur_start, cur_type = None, None
    for idx, tag in enumerate(seq):
        if tag == "O" or not ("-" in tag):
            if cur_type is not None:
                spans.append((cur_start, idx - 1, cur_type))
                cur_start, cur_type = None, None
            continue
        prefix, ent_type = tag.split("-", 1)
        if prefix == "B" or cur_type != ent_type:
            if cur_type is not None:
                spans.append((cur_start, idx - 1, cur_type))
            cur_start, cur_type = idx, ent_type
    if cur_type is not None:
        spans.append((cur_start, len(seq) - 1, cur_type))
    return spans


def analyze_ner_dev_errors(true_seqs: list[list[str]], pred_seqs: list[list[str]]) -> dict:
    """Thong ke dinh luong 4 kieu loi tren tap dev PhoNER (khong in nguyen van cau PhoNER len repo cong khai):
        1. Loi bien thuc the (Boundary error: giao nhau vi tri nhung lech start/end)
        2. Loi nham loai thuc the (Type confusion: dung bien start/end nhung sai nhan thuc the)
        3. Loi bo sot thuc the (Missed entity / False Negative: toan bo span bi du doan la O)
        4. Loi du doan thua & chuoi I- bat hop le (False Positive / Orphan I-tag do thieu CRF)
    """
    exact_match = 0
    boundary_err = 0
    type_err = 0
    missed_err = 0
    spurious_err = 0
    orphan_i_tags = 0
    type_confusion_pairs = Counter()
    boundary_by_type = Counter()

    for t_seq, p_seq in zip(true_seqs, pred_seqs):
        for idx, tag in enumerate(p_seq):
            if tag.startswith("I-"):
                ent = tag[2:]
                prev = p_seq[idx - 1] if idx > 0 else "O"
                if prev not in (f"B-{ent}", f"I-{ent}"):
                    orphan_i_tags += 1

        t_spans = extract_iob2_spans(t_seq)
        p_spans = extract_iob2_spans(p_seq)
        p_by_pos = {(s, e): tp for s, e, tp in p_spans}
        matched_pred = set()

        for s, e, tp in t_spans:
            if (s, e) in p_by_pos:
                matched_pred.add((s, e))
                if p_by_pos[(s, e)] == tp:
                    exact_match += 1
                else:
                    type_err += 1
                    type_confusion_pairs[(tp, p_by_pos[(s, e)])] += 1
            else:
                overlaps = [(ps, pe, ptp) for ps, pe, ptp in p_spans if max(s, ps) <= min(e, pe)]
                if overlaps:
                    boundary_err += 1
                    boundary_by_type[tp] += 1
                    for ps, pe, _ in overlaps:
                        matched_pred.add((ps, pe))
                else:
                    missed_err += 1

        for ps, pe, _ in p_spans:
            if (ps, pe) not in matched_pred:
                spurious_err += 1

    total_gold = exact_match + boundary_err + type_err + missed_err
    print("PHAN TICH DINH LUONG CAC KIEU LOI TREN TAP DEV PHONER (STRICT IOB2):")
    print(f"    - Tong thuc the that tren dev           : {total_gold:,} (khop tuyet doi: {exact_match:,})")
    print(f"    - [Kieu loi 1] Sai bien (Boundary)      : {boundary_err:,} thuc the "
          f"(nhieu nhat o: {', '.join(f'{k}={v}' for k, v in boundary_by_type.most_common(3)) or '—'})")
    print(f"    - [Kieu loi 2] Nham loai (Type conf.)   : {type_err:,} thuc the "
          f"(cap nham: {', '.join(f'{a}->{b} ({c})' for (a, b), c in type_confusion_pairs.most_common(3)) or '—'})")
    print(f"    - [Kieu loi 3] Bo sot (Missed -> 'O')   : {missed_err:,} thuc the")
    print(f"    - [Kieu loi 4] Du doan thua / Orphan I- : {spurious_err:,} span thua | "
          f"{orphan_i_tags:,} vi tri nhan 'I-' mo coi (ko co 'B-' cung loai dung truoc)\n")

    return {
        "total_gold": total_gold,
        "exact_match": exact_match,
        "boundary_err": boundary_err,
        "type_err": type_err,
        "missed_err": missed_err,
        "spurious_err": spurious_err,
        "orphan_i_tags": orphan_i_tags,
    }


@torch.no_grad()
def demo_synthetic_ner_examples(
    model: nn.Module,
    ner_vocab: dict[str, int],
    tags: list[str],
    device,
) -> None:
    #Minh hoa du doan tren cac cau VI DU TU TAO (tuan thu quy dinh khong phat tan du lieu PhoNER)
    synthetic_sentences = [
        ["Bệnh", "nhân", "BN9999", "nam", "45", "tuổi", "điều", "trị", "tại", "Bệnh", "viện",
         "Chợ", "Rẫy", ",", "TP", "Hồ", "Chí", "Minh", "từ", "ngày", "15/08", "."],
        ["Ca", "bệnh", "sốt", ",", "ho", "khan", "sau", "khi", "đi", "chuyến", "bay", "VN123",
         "từ", "Đà", "Nẵng", "về", "Trung", "tâm", "Y", "tế", "quận", "Tân", "Bình", "."],
        ["Bác", "sĩ", "Nguyễn", "Minh", "An", "khám", "cho", "bệnh", "nhân", "nữ", "62", "tuổi",
         "có", "triệu", "chứng", "khó", "thở", "kéo", "dài", "3", "ngày", "."],
    ]

    model.eval()
    print("MINH HOA DU DOAN BIO TREN 3 CAU VI DU TU TAO (KHONG TRICH TU TAP DU LIEU PHONER):")
    for idx, words in enumerate(synthetic_sentences, start=1):
        ids = torch.tensor([encode_tokens(words, ner_vocab)], dtype=torch.long, device=device)
        lengths = torch.tensor([len(words)], dtype=torch.long)
        pred_ids = model(ids, lengths).argmax(dim=-1)[0].cpu().tolist()
        pred_tags = [tags[i] for i in pred_ids]
        formatted = " ".join(f"{w}[{t}]" if t != "O" else w for w, t in zip(words, pred_tags))
        print(f"  Vi du tu tao {idx}: {formatted}")
    print()


def print_cls_comparison_table(records: list[dict], title: str = "BANG SO SANH PHAN LOAI CAM XUC UIT-VSFC") -> None:
    #In bang so sanh cac mo hinh phan loai tren cung bo chia train/dev/test UIT-VSFC
    print(f"=== {title} ===")
    header = (
        f"{'Mo hinh':<30} | {'Tham so':>12} | {'Best Ep':^7} | {'s/ep':>9} | "
        f"{'Dev F1':>8} | {'Test Acc':>8} | {'Test P':>8} | {'Test R':>8} | {'Test F1':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        ep_str = str(r["best_epoch"]) if r.get("best_epoch") is not None else "—"
        sec_str = r.get("sec_note") or (f"{r['sec_per_epoch']:.2f}s" if r.get("sec_per_epoch") is not None else "—")
        dev_f1 = r.get("best_dev_f1", 0.0)
        print(
            f"{r['model']:<30} | {r['params']:>12,} | {ep_str:^7} | {sec_str:>9} | "
            f"{dev_f1:>8.4f} | {r['accuracy']:>8.4f} | {r['precision']:>8.4f} | "
            f"{r['recall']:>8.4f} | {r['macro_f1']:>8.4f}"
        )
    print()


def print_final_submission_table(
    records: list[dict],
    title: str = "BANG TONG HOP KET QUA CAN NOP (LAB 03)",
) -> None:
    #In dung bang tong hop chuan cua Lab 3 (ghi ro loai F1, cau hinh/tham so va giay/epoch)
    print(f"=== {title} ===")
    header = (
        f"{'Mo hinh':<30} | {'Dev F1 tot nhat':>15} | {'Test F1':>9} | "
        f"{'Do do F1':<21} | {'Tham so / Cau hinh':<38} | {'Giay/epoch':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        p_str = r.get("params_note") or f"{r['params']:,} tham so (ep {r.get('best_epoch', '—')})"
        sec_str = r.get("sec_note") or (f"{r['sec_per_epoch']:.2f}s" if r.get("sec_per_epoch") is not None else "—")
        print(
            f"{r['model']:<30} | {r['best_dev_f1']:>15.4f} | {r['test_f1']:>9.4f} | "
            f"{r['task']:<21} | {p_str:<38} | {sec_str:>12}"
        )
    print("Luu y: Macro-F1 o cac dong phan loai UIT-VSFC chi so sanh truc tiep voi nhau; "
          "Entity micro-F1 (strict IOB2) la bai toan gan nhan chuoi tren PhoNER.\n")


def plot_cls_model_comparison(records: list[dict], save_path: str | Path | None = None) -> None:
    #Bieu do cot so sanh Dev macro-F1 va Test macro-F1 giua cac mo hinh tren UIT-VSFC
    if not records:
        return

    names = [r["model"] for r in records]
    dev_f1s = [r["best_dev_f1"] for r in records]
    test_f1s = [r["macro_f1"] for r in records]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.8))
    b1 = ax.bar(x - width / 2, dev_f1s, width, label="Dev macro-F1 (tot nhat)", color="#4299e1")
    b2 = ax.bar(x + width / 2, test_f1s, width, label="Test macro-F1 (chinh)", color="#dd6b20")

    ax.set_xticks(x, names, rotation=15, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("macro-F1")
    ax.set_title("So sanh Dev macro-F1 va Test macro-F1 tren UIT-VSFC", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.4f}", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
