from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

PAD = 0
UNK = 1
PAD_TAG = -100

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(SENTIMENT_LABELS)}

#Danh sach 6 tep du lieu chuan kem SHA-256 checksum theo Lab 3
DATA_FILES = [
    (
        "uit-vsfc/UIT-VSFC-train.json",
        "https://drive.google.com/uc?export=download&id=1eJdEStCQBf3GtbOIQfIR2iS-Wb3nlfJX",
        "e41e65672f2ade65ee8d2d394659a2bfeabbce987f0d2ad9e19998068a78fc1e",
    ),
    (
        "uit-vsfc/UIT-VSFC-dev.json",
        "https://drive.google.com/uc?export=download&id=1UWbzHFeUPu4AuofOEy0Fo0SFotVZHzUe",
        "ab781a38209c17e640e077f83d82ecb9ea4f3469e63fc6ec2a232b0da6fbe823",
    ),
    (
        "uit-vsfc/UIT-VSFC-test.json",
        "https://drive.google.com/uc?export=download&id=1g4imdabNbswIu91K6zKFGalkjDBBqaQq",
        "14a6c5697907084086551a17c76e908dad88f2dc3b8be720c9880c8fc1212aa6",
    ),
    (
        "phoner/train_syllable.json",
        "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/train_syllable.json",
        "fb4097f6ac956a03e484319898db0dafc7283c135cd7888f803ad26a5a253419",
    ),
    (
        "phoner/dev_syllable.json",
        "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/dev_syllable.json",
        "c74393fb5d1547d51ddda175830892ca62a7efa2d93291b21a35cb21bfee612e",
    ),
    (
        "phoner/test_syllable.json",
        "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/test_syllable.json",
        "4b677028d6db5e1dab74d727d88b719454da60cfeaba681d9ee401fc83b04c77",
    ),
]


def ensure_datasets(data_dir: str | Path = "data") -> Path:
    #Kiem tra, tai truc tiep 6 tep UIT-VSFC & PhoNER va xac thuc SHA-256 checksum
    root = Path(data_dir)
    for rel, url, expected in DATA_FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Nguon tai da doi hoac tra ve HTML: {rel}")
            path.write_bytes(content)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum SHA-256 khong khop: {rel}")
    return root


def load_raw_datasets(data_dir: str | Path = "data"):
    """Doc du lieu UIT-VSFC (JSON array) va PhoNER syllable (JSON Lines):
        - UIT-VSFC: chi dung sentence va sentiment (3 lop), khong dung topic
        - PhoNER  : giu nguyen danh sach words va tags da tach san theo syllable
    """
    root = ensure_datasets(data_dir)
    vsfc, ner = {}, {}

    for split in ("train", "dev", "test"):
        vsfc_path = root / "uit-vsfc" / f"UIT-VSFC-{split}.json"
        vsfc[split] = json.loads(vsfc_path.read_text(encoding="utf-8-sig"))
        assert all(r["sentiment"] in LABEL_TO_ID for r in vsfc[split]), f"Nhan la trong UIT-VSFC {split}"

        ner_path = root / "phoner" / f"{split}_syllable.json"
        ner[split] = [
            json.loads(line)
            for line in ner_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        assert all(len(r["words"]) == len(r["tags"]) > 0 for r in ner[split]), f"Lech words/tags trong PhoNER {split}"

    assert [len(vsfc[s]) for s in ("train", "dev", "test")] == [11426, 1583, 3166]
    assert [len(ner[s]) for s in ("train", "dev", "test")] == [5027, 2000, 3000]
    return vsfc, ner


def tokenize_vsfc(text: str) -> list[str]:
    #Tach tu theo khoang trang va dua ve chu thuong cho UIT-VSFC
    return text.lower().split()


def build_vocab(sequences, min_freq: int = 2) -> dict[str, int]:
    #Xay tu dien chi tu tap train (min_freq=2), co dinh <pad>=0 va <unk>=1
    freq = Counter(tok for tokens in sequences for tok in tokens)
    kept = sorted(tok for tok, cnt in freq.items() if cnt >= min_freq)
    return {"<pad>": PAD, "<unk>": UNK, **{tok: i + 2 for i, tok in enumerate(kept)}}


def encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    #Chuyen token chua thay thanh UNK; cau rong van tra [UNK] de length > 0 cho packing
    return [vocab.get(t, UNK) for t in tokens] or [UNK]


def build_cls_rows(rows: list[dict], vocab: dict[str, int], label_to_id: dict[str, int] = LABEL_TO_ID):
    #Ma hoa cap (ids, sentiment_id) cho bai toan phan loai cam xuc UIT-VSFC
    return [(encode_tokens(tokenize_vsfc(r["sentence"]), vocab), label_to_id[r["sentiment"]]) for r in rows]


def collate_cls(batch):
    #Giu lai do dai that truoc khi dem PAD=0 cho batch phan loai van ban
    ids = [torch.tensor(x, dtype=torch.long) for x, _ in batch]
    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    padded_ids = pad_sequence(ids, batch_first=True, padding_value=PAD)
    labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    return padded_ids, lengths, labels


def build_cls_loaders(
    vsfc: dict[str, list[dict]],
    batch_size: int = 32,
    min_freq: int = 2,
    seed: int = 42,
):
    #Tao tu dien tu train va DataLoader dong bo theo seed=42 cho UIT-VSFC
    vocab = build_vocab((tokenize_vsfc(r["sentence"]) for r in vsfc["train"]), min_freq=min_freq)
    assert vocab["<pad>"] == PAD and vocab["<unk>"] == UNK

    train_rows = build_cls_rows(vsfc["train"], vocab)
    dev_rows = build_cls_rows(vsfc["dev"], vocab)
    test_rows = build_cls_rows(vsfc["test"], vocab)

    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_rows,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cls,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=pin,
    )
    dev_dl = DataLoader(
        dev_rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cls,
        pin_memory=pin,
    )
    test_dl = DataLoader(
        test_rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cls,
        pin_memory=pin,
    )
    return train_dl, dev_dl, test_dl, vocab


def build_ner_rows(rows: list[dict], ner_vocab: dict[str, int], tag_to_id: dict[str, int]):
    #Ma hoa cap (word_ids, bio_tag_ids) cho PhoNER, giu nguyen do dai cau (khong cat)
    return [(encode_tokens(r["words"], ner_vocab), [tag_to_id[t] for t in r["tags"]]) for r in rows]


def collate_ner(batch):
    #Padding dong theo cau dai nhat cua batch: words dem PAD=0, tags dem PAD_TAG=-100
    ids = [torch.tensor(x, dtype=torch.long) for x, _ in batch]
    tags = [torch.tensor(y, dtype=torch.long) for _, y in batch]
    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    padded_ids = pad_sequence(ids, batch_first=True, padding_value=PAD)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=PAD_TAG)
    return padded_ids, lengths, padded_tags


def build_ner_loaders(
    ner: dict[str, list[dict]],
    batch_size: int = 32,
    min_freq: int = 2,
    seed: int = 42,
):
    #Xay tu dien va danh sach 21 nhan BIO tu train PhoNER, tao DataLoader dong bo seed=42
    ner_vocab = build_vocab((r["words"] for r in ner["train"]), min_freq=min_freq)
    tags = sorted({tag for row in ner["train"] for tag in row["tags"]})
    tag_to_id = {tag: i for i, tag in enumerate(tags)}

    assert len(tags) == 21, f"So nhan BIO phai la 21, nhan duoc: {len(tags)}"
    assert all(
        tag in tag_to_id
        for split in ("dev", "test")
        for row in ner[split]
        for tag in row["tags"]
    ), "Phat hien nhan BIO o dev/test khong co trong train!"

    train_rows = build_ner_rows(ner["train"], ner_vocab, tag_to_id)
    dev_rows = build_ner_rows(ner["dev"], ner_vocab, tag_to_id)
    test_rows = build_ner_rows(ner["test"], ner_vocab, tag_to_id)

    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_rows,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ner,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=pin,
    )
    dev_dl = DataLoader(
        dev_rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ner,
        pin_memory=pin,
    )
    test_dl = DataLoader(
        test_rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ner,
        pin_memory=pin,
    )
    return train_dl, dev_dl, test_dl, ner_vocab, tags, tag_to_id
