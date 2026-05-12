# %%
import os
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

from huggingface_hub import login
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

argparser = argparse.ArgumentParser(description="CCST with InfoNCE loss on AG News")
argparser.add_argument("--n_soft_tokens", type=int, default=20, help="Number of soft tokens per task/class")
argparser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
argparser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for soft prompt")
argparser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
argparser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = argparser.parse_args()


# %%
HF_TOKEN = ""

hf_token = os.environ.get("HF_TOKEN", None)
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    print("HF_TOKEN 環境変数でログイン完了")
else:
    login()

# %%
MODEL_NAME      = "google/gemma-3-1b-it"
NUM_SOFT_TOKENS = args.n_soft_tokens          # task_soft / per-class soft それぞれのトークン数
NUM_CLASSES     = 4           # AG News のクラス数
BATCH_SIZE      = 4
LR              = args.lr
NUM_EPOCHS      = args.epochs
MAX_TEXT_LEN    = 256         # テキストのmax token長
TRAIN_RATIO     = 0.9         # training set のうち学習に使う割合 (残りをvalidation)
TEMPERATURE     = args.temperature        # InfoNCE temperature
SEED            = args.seed
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

SAVE_DIR = f"ccst_infonce-{LR}-{NUM_SOFT_TOKENS}soft-{TEMPERATURE}temp"
os.makedirs(SAVE_DIR, exist_ok=True)

set_seed(SEED)

print("Configuration:")
for k, v in vars(args).items():
    print(f"  {k}: {v}")

# %%
class AGNewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def load_and_split(train_ratio: float = TRAIN_RATIO, seed: int = SEED):
    """AG News training setをtrain/valに分割して返す。"""
    raw    = load_dataset("ag_news", split="train")
    texts  = raw["text"]
    labels = raw["label"]   # 0: World, 1: Sports, 2: Business, 3: Sci/Tech

    idx = list(range(len(texts)))
    random.seed(seed)
    random.shuffle(idx)

    cut      = int(len(idx) * train_ratio)
    train_ds = AGNewsDataset([texts[i] for i in idx[:cut]], [labels[i] for i in idx[:cut]])
    val_ds   = AGNewsDataset([texts[i] for i in idx[cut:]], [labels[i] for i in idx[cut:]])
    return train_ds, val_ds

def make_collate_fn(tokenizer):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            max_length=MAX_TEXT_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return enc["input_ids"], torch.tensor(labels, dtype=torch.long)
    return collate

# %%
class CCSTSoftPrompt(nn.Module):
    """
    task_soft  : (NUM_SOFT_TOKENS, H)          タスク共通のsoft prompt
    class_soft : (NUM_CLASSES, NUM_SOFT_TOKENS, H)  クラスごとのsoft prompt
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.task_soft  = nn.Parameter(
            torch.randn(NUM_SOFT_TOKENS, hidden_size) * 0.01
        )
        self.class_soft = nn.Parameter(
            torch.randn(NUM_CLASSES, NUM_SOFT_TOKENS, hidden_size) * 0.01
        )

    def get_prefix(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        class_indices : (B,) long tensor
        return        : (B, 2*NUM_SOFT_TOKENS, H)
        """
        B    = class_indices.shape[0]
        task = self.task_soft.unsqueeze(0).expand(B, -1, -1)   # (B, S, H)
        cls  = self.class_soft[class_indices]                   # (B, S, H)
        return torch.cat([task, cls], dim=1)                    # (B, 2S, H)

# %%
def pool_text_hidden(
    model,
    prefix_embeds,       # (B, P, H) or None
    text_ids: torch.Tensor,   # (B, T)
    pad_id: int,
) -> torch.Tensor:
    """
    text positions（prefix を除いた部分）の hidden states を mean-pool して返す。
    prefix_embeds が None のときはテキスト単体で forward する。
    """
    # テキストのtoken embeddingを取得
    text_embeds = model.get_input_embeddings()(text_ids)    # (B, T, H)

    if prefix_embeds is not None:
        prefix_embeds = prefix_embeds.to(text_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
        prefix_len    = prefix_embeds.shape[1]
    else:
        inputs_embeds = text_embeds
        prefix_len    = 0

    out    = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
    hidden = out.hidden_states[-1]                          # (B, P+T, H)

    # text positions だけスライス
    text_hidden = hidden[:, prefix_len:, :]                 # (B, T, H)

    # padding を除いてmean pool
    mask   = (text_ids != pad_id).float().unsqueeze(-1)     # (B, T, 1)
    pooled = (text_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled                                           # (B, H)

# %%
def infonce_loss(
    anchor:    torch.Tensor,        # (B, H)
    positive:  torch.Tensor,        # (B, H)
    negatives: list[torch.Tensor],  # list of (B, H)
    tau: float = TEMPERATURE,
) -> torch.Tensor:
    """
    anchor と positive の類似度を最大化し、
    anchor と negatives の類似度を最小化する InfoNCE loss。
    """
    anchor   = F.normalize(anchor,   dim=-1)
    positive = F.normalize(positive, dim=-1)
    negs     = [F.normalize(n, dim=-1) for n in negatives]

    sim_pos  = (anchor * positive).sum(-1, keepdim=True) / tau         # (B, 1)
    sim_negs = torch.cat(
        [(anchor * n).sum(-1, keepdim=True) / tau for n in negs], dim=-1
    )                                                                   # (B, K)

    # logits の先頭が positive → label = 0 で cross entropy
    logits = torch.cat([sim_pos, sim_negs], dim=-1)                    # (B, 1+K)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

# %%
@torch.no_grad()
def evaluate(model, soft_prompt, loader, pad_id: int, device: str) -> float:
    """
    各クラスの anchor 表現と positive 表現のコサイン類似度が最大のクラスを予測し、
    accuracy を返す。
    """
    model.eval()
    correct = total = 0

    for text_ids, labels in tqdm(loader, desc="  eval", leave=False):
        text_ids = text_ids.to(device)
        labels   = labels.to(device)
        B        = text_ids.shape[0]

        # positive: text-only representation
        pos_rep = F.normalize(
            pool_text_hidden(model, None, text_ids, pad_id), dim=-1
        )   # (B, H)

        # 各クラスの anchor 表現との類似度
        sims = []
        for c in range(NUM_CLASSES):
            cidx   = torch.full((B,), c, dtype=torch.long, device=device)
            prefix = soft_prompt.get_prefix(cidx)
            anc    = F.normalize(
                pool_text_hidden(model, prefix, text_ids, pad_id), dim=-1
            )
            sims.append((anc * pos_rep).sum(-1))   # (B,)

        preds = torch.stack(sims, dim=-1).argmax(-1)    # (B,)
        correct += (preds == labels).sum().item()
        total   += B

    return correct / total

# %%
print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_NAME}")

# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16
).to(DEVICE)

# LM のパラメータを全て固定
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

hidden_size = model.config.hidden_size
print(f"Hidden size : {hidden_size}")

# Soft Prompt（これだけが学習対象）
soft_prompt = CCSTSoftPrompt(hidden_size).to(DEVICE)
optimizer   = torch.optim.AdamW(soft_prompt.parameters(), lr=LR)

# Dataset
print("Loading AG News …")
train_ds, val_ds = load_and_split()
print(f"  train={len(train_ds):,}, val={len(val_ds):,}")

collate_fn = make_collate_fn(tokenizer)
train_ldr  = DataLoader(train_ds, batch_size=BATCH_SIZE,
                        shuffle=True,  collate_fn=collate_fn)
val_ldr    = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

# %%
best_val_acc = 0.0
for epoch in range(1, NUM_EPOCHS + 1):
    soft_prompt.train()
    running_loss = 0.0
    n_batches    = 0

    for text_ids, labels in tqdm(train_ldr, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        text_ids = text_ids.to(DEVICE)
        labels   = labels.to(DEVICE)
        B        = text_ids.shape[0]

        optimizer.zero_grad()

        # positive: text-only representation（LMは固定なのでgrad不要）
        with torch.no_grad():
            pos_rep = pool_text_hidden(model, None, text_ids, pad_id)

        # 全クラス分の anchor 表現を計算（soft promptを通じてgradが流れる）
        class_reps = []
        for c in range(NUM_CLASSES):
            cidx   = torch.full((B,), c, dtype=torch.long, device=DEVICE)
            prefix = soft_prompt.get_prefix(cidx)
            rep    = pool_text_hidden(model, prefix, text_ids, pad_id)
            class_reps.append(rep)

        # クラスごとにInfoNCE lossを計算して合算
        loss    = torch.tensor(0.0, device=DEVICE)
        n_terms = 0

        for c in range(NUM_CLASSES):
            mask = (labels == c)
            if mask.sum() == 0:
                continue

            anchor    = class_reps[c][mask]
            positive  = pos_rep[mask].detach()   # LM固定なので detach
            negatives = [class_reps[c_][mask]
                            for c_ in range(NUM_CLASSES) if c_ != c]

            loss    = loss + infonce_loss(anchor, positive, negatives)
            n_terms += 1

        if n_terms > 0:
            loss = loss / n_terms
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

    avg_loss = running_loss / max(n_batches, 1)
    val_acc  = evaluate(model, soft_prompt, val_ldr, pad_id, DEVICE)
    print(f"  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # モデルの保存（ベストモデルのみ）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(soft_prompt.state_dict(), f"{SAVE_DIR}/ccst_infonce.pt")
        print(f"Saved: {SAVE_DIR}/ccst_infonce.pt")


