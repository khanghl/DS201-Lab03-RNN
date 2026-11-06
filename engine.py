from collections import Counter
import copy
import csv
from pathlib import Path
import time

from seqeval.metrics import (
    f1_score as entity_f1_score,
    precision_score as entity_precision_score,
    recall_score as entity_recall_score,
)
from seqeval.scheme import IOB2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
import torch
import torch.nn as nn


#Baseline cho UIT-VSFC: Majority class va TF-IDF (1-2 gram) + LogisticRegression
def fit_tfidf_baseline(vsfc_train: list[dict], vsfc_dev: list[dict], label_to_id: dict[str, int]):
    train_text = [r["sentence"] for r in vsfc_train]
    dev_text = [r["sentence"] for r in vsfc_dev]
    train_y = [label_to_id[r["sentiment"]] for r in vsfc_train]
    dev_y = [label_to_id[r["sentiment"]] for r in vsfc_dev]

    majority_cls = Counter(train_y).most_common(1)[0][0]
    maj_preds = [majority_cls] * len(dev_y)
    maj_dev_f1 = f1_score(dev_y, maj_preds, labels=[0, 1, 2], average="macro", zero_division=0)
    maj_dev_acc = accuracy_score(dev_y, maj_preds)

    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r"(?u)\b\w+\b"),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    t0 = time.perf_counter()
    pipeline.fit(train_text, train_y)
    fit_sec = time.perf_counter() - t0

    dev_pred = pipeline.predict(dev_text)
    dev_f1 = f1_score(dev_y, dev_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    dev_acc = accuracy_score(dev_y, dev_pred)

    vec = pipeline.named_steps["tfidfvectorizer"]
    clf = pipeline.named_steps["logisticregression"]
    n_features = len(vec.vocabulary_)
    n_weights = int(clf.coef_.size + clf.intercept_.size)

    summary = {
        "majority_cls": majority_cls,
        "majority_dev_f1": maj_dev_f1,
        "majority_dev_acc": maj_dev_acc,
        "best_dev_f1": dev_f1,
        "dev_acc": dev_acc,
        "dev_y": dev_y,
        "dev_pred": dev_pred.tolist(),
        "fit_sec": fit_sec,
        "n_features": n_features,
        "params": n_weights,
        "config_str": f"TF-IDF(1-2g, {n_features:,} feat) + LogReg({n_weights:,} w)",
    }
    return pipeline, summary


#Vong lap huan luyen & danh gia cho phan loai cam xuc UIT-VSFC (Bai 1, Bai 2, Bai 4*)
def train_one_epoch_cls(model, loader, criterion, optimizer, device, max_norm: float = 1.0):
    model.train() #bat dropout
    total, correct, running_loss = 0, 0, 0.0
    for ids, lengths, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm) #chan chuan gradient = 1.0
        optimizer.step()

        bs = labels.numel()
        running_loss += loss.item() * bs
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += bs
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_cls(model, loader, criterion, device, num_classes: int = 3):
    model.eval() #tat dropout, danh gia voi no_grad
    total, correct, running_loss = 0, 0, 0.0
    y_true, y_pred = [], []
    for ids, lengths, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids, lengths)
        bs = labels.numel()
        running_loss += criterion(logits, labels).item() * bs
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += bs
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    return running_loss / total, correct / total, macro_f1, y_true, y_pred


def fit_cls(
    model,
    train_dl,
    dev_dl,
    epochs: int,
    lr: float,
    device,
    out_dir: str | Path,
    classes: list[str],
    max_norm: float = 1.0,
    seed: int = 42,
    criterion=None,
    optimizer=None,
):
    """Huan luyen mo hinh phan loai van ban:
        - Khoi phuc generator cua train_dl theo seed de moi mo hinh nhan cung thu tu batch
        - Clip gradient norm = 1.0 sau backward, truoc step
        - Giu checkpoint best.pt co macro-F1 tren tap dev cao nhat va luu history.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    if getattr(train_dl, "generator", None) is not None:
        train_dl.generator.manual_seed(seed)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)

    history = []
    best_f1, best_epoch = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    ckpt_path = out_dir / "best.pt"

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch_cls(model, train_dl, criterion, optimizer, device, max_norm=max_norm)
        dv_loss, dv_acc, dv_f1, _, _ = evaluate_cls(model, dev_dl, criterion, device, num_classes=len(classes))
        ep_sec = time.perf_counter() - t0

        history.append(
            dict(
                epoch=ep,
                train_loss=tr_loss,
                train_acc=tr_acc,
                dev_loss=dv_loss,
                dev_acc=dv_acc,
                dev_f1=dv_f1,
                sec=ep_sec,
            )
        )

        mark = ""
        if dv_f1 > best_f1: #chon checkpoint theo macro-F1 tren dev
            best_f1 = dv_f1
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model": best_state,
                    "classes": classes,
                    "epoch": ep,
                    "dev_f1": dv_f1,
                },
                ckpt_path,
            )
            mark = "  *"

        print(
            f"epoch {ep:>2}/{epochs:<2}  train {tr_loss:.4f}/{tr_acc:.4f}"
            f"   dev {dv_loss:.4f}/{dv_acc:.4f}  macro-F1 {dv_f1:.4f}"
            f"   ({ep_sec:.2f}s){mark}"
        )

    if history:
        with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    avg_sec = sum(h["sec"] for h in history) / max(len(history), 1)
    model.load_state_dict(best_state)
    model.eval()
    print(
        f"-> best dev macro-F1: {best_f1:.4f} (epoch {best_epoch}) | "
        f"trung binh: {avg_sec:.2f}s/epoch | da luu: {ckpt_path}\n"
    )
    summary = {
        "best_epoch": best_epoch,
        "best_dev_f1": best_f1,
        "sec_per_epoch": avg_sec,
        "ckpt_path": str(ckpt_path),
    }
    return history, summary


#Vong lap huan luyen & danh gia cho gan nhan thuc the PhoNER (Bai 3)
def train_one_epoch_ner(
    model,
    loader,
    criterion,
    optimizer,
    device,
    pad_tag: int = -100,
    max_norm: float = 1.0,
):
    model.train()
    running_loss, total_tokens = 0.0, 0
    for ids, lengths, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, lengths)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        valid_tokens = int((labels != pad_tag).sum().item())
        running_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
    return running_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_ner(
    model,
    loader,
    criterion,
    tags: list[str],
    device,
    pad_tag: int = -100,
):
    #Cat du doan va nhan that theo do dai that (bo PAD_TAG) truoc khi cham seqeval strict IOB2
    model.eval()
    running_loss, total_tokens = 0.0, 0
    true_seqs, pred_seqs = [], []

    for ids, lengths, labels in loader:
        ids_dev, labels_dev = ids.to(device), labels.to(device)
        logits = model(ids_dev, lengths)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels_dev.reshape(-1))

        valid_tokens = int((labels_dev != pad_tag).sum().item())
        running_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

        preds = logits.argmax(dim=-1).cpu()
        for actual, guessed, length in zip(labels, preds, lengths):
            n = int(length)
            true_seqs.append([tags[int(i)] for i in actual[:n]])
            pred_seqs.append([tags[int(i)] for i in guessed[:n]])

    ent_prec = entity_precision_score(true_seqs, pred_seqs, mode="strict", scheme=IOB2, zero_division=0)
    ent_rec = entity_recall_score(true_seqs, pred_seqs, mode="strict", scheme=IOB2, zero_division=0)
    ent_f1 = entity_f1_score(true_seqs, pred_seqs, mode="strict", scheme=IOB2, zero_division=0)
    return running_loss / max(total_tokens, 1), ent_prec, ent_rec, ent_f1, true_seqs, pred_seqs


def fit_ner(
    model,
    train_dl,
    dev_dl,
    tags: list[str],
    epochs: int,
    lr: float,
    device,
    out_dir: str | Path,
    pad_tag: int = -100,
    max_norm: float = 1.0,
    seed: int = 42,
    criterion=None,
    optimizer=None,
):
    """Huan luyen Encoder BiLSTM cho PhoNER:
        - CrossEntropyLoss(ignore_index=-100) bo qua vi tri dem
        - Giu checkpoint best.pt co entity micro-F1 (strict IOB2) cao nhat tren dev
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    if getattr(train_dl, "generator", None) is not None:
        train_dl.generator.manual_seed(seed)

    if criterion is None:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_tag)
    if optimizer is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)

    history = []
    best_f1, best_epoch = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    ckpt_path = out_dir / "best.pt"

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss = train_one_epoch_ner(
            model, train_dl, criterion, optimizer, device, pad_tag=pad_tag, max_norm=max_norm
        )
        dv_loss, dv_prec, dv_rec, dv_f1, _, _ = evaluate_ner(
            model, dev_dl, criterion, tags, device, pad_tag=pad_tag
        )
        ep_sec = time.perf_counter() - t0

        history.append(
            dict(
                epoch=ep,
                train_loss=tr_loss,
                dev_loss=dv_loss,
                dev_prec=dv_prec,
                dev_rec=dv_rec,
                dev_f1=dv_f1,
                sec=ep_sec,
            )
        )

        mark = ""
        if dv_f1 > best_f1: #chon checkpoint theo entity micro-F1 tren dev
            best_f1 = dv_f1
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model": best_state,
                    "tags": tags,
                    "epoch": ep,
                    "dev_entity_f1": dv_f1,
                },
                ckpt_path,
            )
            mark = "  *"

        print(
            f"epoch {ep:>2}/{epochs:<2}  train_loss {tr_loss:.4f}"
            f"   dev_loss {dv_loss:.4f}  entity-F1 {dv_f1:.4f} (P={dv_prec:.4f}, R={dv_rec:.4f})"
            f"   ({ep_sec:.2f}s){mark}"
        )

    if history:
        with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    avg_sec = sum(h["sec"] for h in history) / max(len(history), 1)
    model.load_state_dict(best_state)
    model.eval()
    print(
        f"-> best dev entity-F1 (strict IOB2): {best_f1:.4f} (epoch {best_epoch}) | "
        f"trung binh: {avg_sec:.2f}s/epoch | da luu: {ckpt_path}\n"
    )
    summary = {
        "best_epoch": best_epoch,
        "best_dev_f1": best_f1,
        "sec_per_epoch": avg_sec,
        "ckpt_path": str(ckpt_path),
    }
    return history, summary
