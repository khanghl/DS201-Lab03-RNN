import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def n_params(module: nn.Module) -> int:
    #Tong so tham so cua module
    return sum(p.numel() for p in module.parameters())


def n_trainable_params(module: nn.Module) -> int:
    #Tong so tham so co requires_grad=True cua module
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def rnn_layer_params_formula(
    kind: str,
    in_dim: int,
    hidden_size: int,
    bidirectional: bool = False,
) -> int:
    #Cong thuc tham so 1 tang hoi quy PyTorch (gom bias_ih va bias_hh): g * H * (in_dim + H + 2) * num_dir
    gates = {"RNN": 1, "GRU": 3, "LSTM": 4}[kind]
    num_dir = 2 if bidirectional else 1
    return num_dir * gates * hidden_size * (in_dim + hidden_size + 2)


#Bai 1, Bai 2 & Bai 4*: Mang hoi quy phan loai cam xuc UIT-VSFC (LSTM / GRU nhieu lop + packing)
class TextClassifier(nn.Module):
    """Mo hinh phan loai van ban hoi quy tren UIT-VSFC (3 lop sentiment):
        - Bai 1 : kind='LSTM', emb_dim=300, hidden_size=256, num_layers=5, dropout=0.3
        - Bai 2 : kind='GRU',  emb_dim=300, hidden_size=256, num_layers=5, dropout=0.3
        - Bai 4*: kind='LSTM', num_layers in {1, 2, 5} de khao sat anh huong do sau
    """

    def __init__(
        self,
        vocab_size: int,
        kind: str = "LSTM",
        emb_dim: int = 300,
        hidden_size: int = 256,
        num_layers: int = 5,
        num_classes: int = 3,
        dropout: float = 0.3,
        pad_idx: int = 0,
        bidirectional: bool = False,
    ):
        super().__init__()
        if kind not in ("RNN", "LSTM", "GRU"):
            raise ValueError(f"kind phai thuoc ('RNN', 'LSTM', 'GRU'), nhan duoc: {kind!r}")

        self.kind = kind
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0, #dropout giua cac tang hoi quy khi layers > 1
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes) #tra logits tho truc tiep cho CrossEntropyLoss

    def inspect_layers(self):
        #Doi chieu tham so tung thanh phan giua cong thuc ly thuyet va PyTorch thuc te
        rows = []
        emb_expected = self.emb.num_embeddings * self.emb_dim
        rows.append(("Embedding", emb_expected, n_params(self.emb)))

        num_dir = 2 if self.bidirectional else 1
        for l_idx in range(self.num_layers):
            in_d = self.emb_dim if l_idx == 0 else self.hidden_size * num_dir
            exp_l = rnn_layer_params_formula(self.kind, in_d, self.hidden_size, self.bidirectional)
            act_l = sum(
                p.numel()
                for name, p in self.rnn.named_parameters()
                if name.endswith(f"_l{l_idx}") or name.endswith(f"_l{l_idx}_reverse")
            )
            rows.append((f"{self.kind} layer {l_idx + 1}", exp_l, act_l))

        fc_in = self.hidden_size * num_dir
        fc_expected = fc_in * self.num_classes + self.num_classes
        rows.append(("Linear (fc)", fc_expected, n_params(self.fc)))
        return rows

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.emb(ids) #[B, L, E]
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, state = self.rnn(packed)
        h = state[0] if self.kind == "LSTM" else state #[num_layers * num_dir, B, H]
        vec = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.fc(self.drop(vec)) #[B, num_classes]


#Bai 3: Encoder BiLSTM 5 lop gan nhan thuc the BIO tren PhoNER_COVID19
class TokenTagger(nn.Module):
    """Encoder BiLSTM nhieu lop gan nhan chuoi BIO cho bai toan NER (PhoNER syllable):
        - Moi vi tri token co bieu dien 2*hidden_size = 512 chieu (ghep xuoi + nguoc)
        - Dung pack_padded_sequence & pad_packed_sequence(total_length=ids.size(1))
        - Tang Linear(512, 21) du doan logits BIO cho tung token doc lap
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int = 21,
        emb_dim: int = 300,
        hidden_size: int = 256,
        num_layers: int = 5,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tags = num_tags

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, num_tags) #Linear(512, 21)

    def inspect_layers(self):
        #Doi chieu tham so tung tang cua Encoder BiLSTM 5 lop + Linear(512, 21)
        rows = []
        emb_expected = self.emb.num_embeddings * self.emb_dim
        rows.append(("Embedding", emb_expected, n_params(self.emb)))

        for l_idx in range(self.num_layers):
            in_d = self.emb_dim if l_idx == 0 else 2 * self.hidden_size
            exp_l = rnn_layer_params_formula("LSTM", in_d, self.hidden_size, bidirectional=True)
            act_l = sum(
                p.numel()
                for name, p in self.rnn.named_parameters()
                if name.endswith(f"_l{l_idx}") or name.endswith(f"_l{l_idx}_reverse")
            )
            rows.append((f"BiLSTM layer {l_idx + 1}", exp_l, act_l))

        fc_expected = (2 * self.hidden_size) * self.num_tags + self.num_tags
        rows.append(("Linear (512->21)", fc_expected, n_params(self.fc)))
        return rows

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            self.emb(ids),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(
            output,
            batch_first=True,
            total_length=ids.size(1), #khoi phuc chieu dai bang batch padded de khop nhan
        )
        return self.fc(self.drop(output)) #[B, L, num_tags]


if __name__ == "__main__":
    #Kiem tra nhanh shape va so tham so cua 3 kien truc chinh
    x_demo = torch.randint(2, 100, (4, 12))
    len_demo = torch.tensor([12, 9, 7, 5])

    m_lstm = TextClassifier(vocab_size=1535, kind="LSTM", emb_dim=300, hidden_size=256, num_layers=5)
    print(f"LSTM 5 lop (V=1535)      : logits={tuple(m_lstm(x_demo, len_demo).shape)} | params={n_params(m_lstm):,}")

    m_gru = TextClassifier(vocab_size=1535, kind="GRU", emb_dim=300, hidden_size=256, num_layers=5)
    print(f"GRU 5 lop  (V=1535)      : logits={tuple(m_gru(x_demo, len_demo).shape)} | params={n_params(m_gru):,}")

    m_ner = TokenTagger(vocab_size=5000, num_tags=21, emb_dim=300, hidden_size=256, num_layers=5)
    print(f"BiLSTM NER 5 lop (V=5000): logits={tuple(m_ner(x_demo, len_demo).shape)} | params={n_params(m_ner):,}")
