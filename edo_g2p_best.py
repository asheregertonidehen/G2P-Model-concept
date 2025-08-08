#!/usr/bin/env python3
"""
edo_g2p_best.py

Optimized attention-based seq2seq Grapheme-to-Phoneme model for Edo.
Dataset format: edo_word|edo_IPA (one entry per line, UTF-8)

Usage:
  python edo_g2p_best.py train
  python edo_g2p_best.py predict
  python edo_g2p_best.py train --epochs 50 --batch 64 --beam 5
"""

import argparse
import os
import time
import random
from typing import List, Dict
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# DEFAULT CONFIG (tweakable)
# ----------------------------
DEFAULTS = {
    "data_file": "new_metadata.txt",
    "checkpoint": "edo_g2p_best_ckpt.pt",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "emb_size": 192,
    "hidden_size": 256,        # optimized as recommended
    "num_layers": 1,
    "batch_size": 64,
    "epochs": 50,              # recommended for best accuracy
    "lr": 1e-3,
    "patience": 8,
    "clip": 1.0,
    "tf_start": 1.0,
    "tf_end": 0.3,
    "valid_ratio": 0.1,
    "beam_width": 5,
    "max_pred_len": 120,
    "print_every": 50,
    "weight_decay": 1e-6,
    "use_amp": True,           # use mixed precision if GPU available
}
# ----------------------------

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_line(ln: str) -> str:
    return ln.strip()

def edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb+1):
            cur = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j-1], dp[j])
            prev = cur
    return dp[lb]

def cer(pred: str, gold: str) -> float:
    if len(gold) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gold) / len(gold)

# ----------------------------
# Dataset
# ----------------------------
class G2PDataset(Dataset):
    def __init__(self, filepath: str, build_vocab: bool = True,
                 char2idx_in: Dict[str,int]=None, char2idx_out: Dict[str,int]=None):
        pairs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for ln in f:
                ln = normalize_line(ln)
                if "|" not in ln: continue
                a,b = ln.split("|",1)
                a = a.strip()
                b = b.strip()
                if not a or not b: continue
                pairs.append((list(a), list(b)))
        self.pairs = pairs

        if build_vocab:
            chars_in = sorted({c for w,_ in pairs for c in w})
            chars_out = sorted({c for _,t in pairs for c in t})
            self.char2idx_in = {PAD:0, SOS:1, EOS:2, UNK:3}
            for i,ch in enumerate(chars_in, start=len(self.char2idx_in)):
                self.char2idx_in[ch] = i
            self.char2idx_out = {PAD:0, SOS:1, EOS:2, UNK:3}
            for i,ch in enumerate(chars_out, start=len(self.char2idx_out)):
                self.char2idx_out[ch] = i
        else:
            self.char2idx_in = char2idx_in
            self.char2idx_out = char2idx_out

        self.idx2char_out = {i:c for c,i in self.char2idx_out.items()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_idx = [ self.char2idx_in.get(ch, self.char2idx_in[UNK]) for ch in src ]
        tgt_idx = [ self.char2idx_out[SOS] ] + [ self.char2idx_out.get(ch, self.char2idx_out[UNK]) for ch in tgt ] + [ self.char2idx_out[EOS] ]
        return torch.tensor(src_idx, dtype=torch.long), torch.tensor(tgt_idx, dtype=torch.long), "".join(src), "".join(tgt)

def collate_fn(batch):
    srcs, tgts, src_strs, tgt_strs = zip(*batch)
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    src_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return src_padded, tgt_padded, src_lens, tgt_lens, src_strs, tgt_strs

# ----------------------------
# Model: Encoder / Attention / Decoder
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim*2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,S,2*hid)
        # make initial decoder hidden by combining last forward & backward
        # h: (num_layers*2, B, hid)
        # take last layer's forward & backward
        fwd = h[-2,:,:]
        bwd = h[-1,:,:]
        h_cat = torch.cat((fwd, bwd), dim=1)  # (B, hid*2)
        h0 = torch.tanh(self.fc(h_cat))       # (B, hid)
        # create (n_layers, B, hid) hidden for decoder (repeat h0)
        dec_h0 = h0.unsqueeze(0)  # (1,B,hid)
        dec_c0 = torch.zeros_like(dec_h0)
        return outputs, (dec_h0, dec_c0)

class BahdanauAttention(nn.Module):
    def __init__(self, dec_hid_dim, enc_hid_dim):
        super().__init__()
        self.attn = nn.Linear(dec_hid_dim + enc_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: (B, hid)
        B, S, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, S, 1)  # (B, S, hid)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (B,S,dec_hid)
        scores = self.v(energy).squeeze(2)  # (B,S)
        scores = scores.masked_fill(mask == 0, -1e9)
        return torch.softmax(scores, dim=1)  # (B,S)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, enc_hid_dim, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, hid_dim, batch_first=True)
        self.attention = BahdanauAttention(hid_dim, enc_hid_dim)
        self.fc_out = nn.Linear(hid_dim + enc_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, cell, encoder_outputs, mask):
        # input_tok: (B,) ids
        input_tok = input_tok.unsqueeze(1)  # (B,1)
        embedded = self.dropout(self.embedding(input_tok))  # (B,1,emb)
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs, mask)  # (B,S)
        attn_weights = attn_weights.unsqueeze(1)  # (B,1,S)
        context = torch.bmm(attn_weights, encoder_outputs)  # (B,1,enc_hid)
        rnn_input = torch.cat((embedded, context), dim=2)  # (B,1, emb+enc_hid)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = output.squeeze(1)        # (B,hid)
        context = context.squeeze(1)      # (B,enc_hid)
        embedded = embedded.squeeze(1)    # (B,emb)
        pred_in = torch.cat((output, context, embedded), dim=1)  # (B, hid+enc_hid+emb)
        prediction = self.fc_out(pred_in)  # (B, output_dim)
        return prediction, hidden, cell, attn_weights.squeeze(1)

# ----------------------------
# Seq2Seq wrapper + beam search
# ----------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device, char2idx_out):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_token = char2idx_out[SOS]
        self.eos_token = char2idx_out[EOS]

    def create_mask(self, src):
        return (src != 0).to(self.device)

    def forward(self, src, src_len, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        encoder_outputs, (hidden, cell) = self.encoder(src, src_len)
        enc_hid_dim = encoder_outputs.size(2)
        max_len = trg.size(1) if trg is not None else DEFAULTS["max_pred_len"]
        outputs = torch.zeros(batch_size, max_len, self.decoder.output_dim).to(self.device)
        mask = self.create_mask(src)
        input_tok = torch.tensor([self.sos_token] * batch_size, dtype=torch.long, device=self.device)
        for t in range(0, max_len):
            preds, hidden, cell, attn = self.decoder(input_tok, hidden, cell, encoder_outputs, mask)
            outputs[:, t, :] = preds
            top1 = preds.argmax(1)
            if trg is not None and random.random() < teacher_forcing_ratio:
                input_tok = trg[:, t]
            else:
                input_tok = top1
        return outputs

# Beam node helper for single-sentence beam search
class BeamNode:
    def __init__(self, tokens: List[int], logprob: float, hidden, cell):
        self.tokens = tokens
        self.logprob = logprob
        self.hidden = hidden
        self.cell = cell

def beam_search_decode(model: Seq2Seq, src_tensor: torch.LongTensor, src_len: List[int],
                       beam_width:int=5, max_len:int=100, idx2char_out=None) -> str:
    model.eval()
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor, src_len)
        mask = model.create_mask(src_tensor)
        B = src_tensor.size(0)
        assert B == 1, "Beam implemented for single example only"
        init = BeamNode(tokens=[decoder_start_token], logprob=0.0, hidden=hidden, cell=cell)
        nodes = [init]
        completed = []
        for _ in range(max_len):
            new_nodes = []
            for node in nodes:
                last = node.tokens[-1]
                if last == decoder_end_token:
                    completed.append(node)
                    continue
                last_tok = torch.tensor([last], dtype=torch.long, device=model.device)
                preds, nh, nc, attn = model.decoder(last_tok, node.hidden, node.cell, encoder_outputs, mask)
                log_probs = torch.log_softmax(preds, dim=1).squeeze(0)  # (V,)
                topk = torch.topk(log_probs, k=min(beam_width, log_probs.size(0)))
                for lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    new_tokens = node.tokens + [int(idx)]
                    new_nodes.append(BeamNode(new_tokens, node.logprob + float(lp), nh, nc))
            nodes = sorted(new_nodes, key=lambda n: n.logprob, reverse=True)[:beam_width]
            if not nodes:
                break
            if len(completed) >= beam_width:
                break
        completed += nodes
        completed = sorted(completed, key=lambda n: n.logprob / max(1, len(n.tokens)), reverse=True)
        best = completed[0]
        out_chars = []
        for tok in best.tokens:
            if tok in (decoder_start_token, decoder_end_token, char2idx_out_global[PAD]):
                continue
            out_chars.append(idx2char_out.get(tok, ""))
        return "".join(out_chars)

# ----------------------------
# Training / eval loops
# ----------------------------
def train_epoch(model, loader, optimizer, criterion, epoch, tf_ratio, scaler=None):
    model.train()
    total_loss = 0.0
    for i, (src, trg, src_lens, trg_lens, _, _) in enumerate(loader):
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(src, src_lens, trg=trg, teacher_forcing_ratio=tf_ratio)
                B, T, V = outputs.shape
                loss = criterion(outputs.view(-1, V), trg[:, :T].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(src, src_lens, trg=trg, teacher_forcing_ratio=tf_ratio)
            B, T, V = outputs.shape
            loss = criterion(outputs.view(-1, V), trg[:, :T].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            optimizer.step()
        total_loss += loss.item()
        if (i+1) % cfg["print_every"] == 0:
            print(f"  batch {i+1}/{len(loader)} - avg loss: {total_loss/(i+1):.4f}")
    return total_loss / len(loader)

def evaluate(model, loader, criterion, idx2char_out):
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    n = 0
    with torch.no_grad():
        for src, trg, src_lens, trg_lens, src_strs, tgt_strs in loader:
            src = src.to(device)
            trg = trg.to(device)
            outputs = model(src, src_lens, trg=trg, teacher_forcing_ratio=0.0)
            B, T, V = outputs.shape
            loss = criterion(outputs.view(-1, V), trg[:, :T].reshape(-1))
            total_loss += loss.item()
            preds = outputs.argmax(dim=2)
            for b in range(B):
                pred_tokens = preds[b].tolist()
                if 2 in pred_tokens:
                    idx = pred_tokens.index(2)
                    pred_tokens = pred_tokens[:idx]
                if len(pred_tokens)>0 and pred_tokens[0]==1:
                    pred_tokens = pred_tokens[1:]
                pred_chars = "".join([idx2char_out.get(tok, "") for tok in pred_tokens if tok not in (0,1,2,3)])
                gold = tgt_strs[b]
                total_cer += cer(pred_chars, gold)
                n += 1
    return total_loss / len(loader), (total_cer / n if n>0 else 0.0)

# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train","predict"])
    p.add_argument("--data", default=DEFAULTS["data_file"])
    p.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--beam", type=int, default=DEFAULTS["beam_width"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = DEFAULTS.copy()
    cfg["data_file"] = args.data
    cfg["checkpoint"] = args.checkpoint
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch
    cfg["beam_width"] = args.beam
    cfg["lr"] = args.lr
    cfg["seed"] = args.seed

    device = torch.device(cfg["device"])
    set_seed(cfg["seed"])
    print("Device:", device)

    if args.mode == "predict":
        if not os.path.exists(cfg["checkpoint"]):
            raise SystemExit(f"No checkpoint found at {cfg['checkpoint']}. Train first.")
        ck = torch.load(cfg["checkpoint"], map_location=device)
        char2idx_in = ck["char2idx_in"]
        char2idx_out = ck["char2idx_out"]
        idx2char_out = {i:c for c,i in char2idx_out.items()}
        # global references for beam decode
        char2idx_out_global = char2idx_out
        decoder_start_token = char2idx_out[SOS]
        decoder_end_token = char2idx_out[EOS]
        # build model
        input_dim = len(char2idx_in)
        output_dim = len(char2idx_out)
        encoder = Encoder(input_dim, cfg["emb_size"], cfg["hidden_size"], n_layers=cfg["num_layers"]).to(device)
        decoder = Decoder(output_dim, cfg["emb_size"], cfg["hidden_size"], enc_hid_dim=cfg["hidden_size"]*2).to(device)
        model = Seq2Seq(encoder, decoder, device, char2idx_out).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        print("Loaded model. Enter Edo words (type 'quit' to exit).")
        while True:
            w = input("Edo word: ").strip()
            if not w or w.lower()=="quit":
                break
            seq = [ char2idx_in.get(ch, char2idx_in[UNK]) for ch in list(w) ]
            src = torch.tensor([seq], dtype=torch.long, device=device)
            src_len = [len(seq)]
            # set globals used inside beam search
            decoder_start_token = char2idx_out[SOS]
            decoder_end_token = char2idx_out[EOS]
            # helper global map used in beam_search function
            char2idx_out_global = char2idx_out
            pred = beam_search_decode(model, src, src_len, beam_width=cfg["beam_width"],
                                      max_len=cfg["max_pred_len"], idx2char_out=idx2char_out)
            print("Predicted IPA:", pred)
        raise SystemExit(0)

    # TRAIN path
    print("Loading and preparing dataset...")
    ds_all = G2PDataset(cfg["data_file"], build_vocab=True)
    total = len(ds_all)
    indices = list(range(total))
    random.shuffle(indices)
    split = int(total * (1 - cfg["valid_ratio"]))
    train_idx, val_idx = indices[:split], indices[split:]
    train_pairs = [ds_all.pairs[i] for i in train_idx]
    val_pairs = [ds_all.pairs[i] for i in val_idx]

    train_ds = G2PDataset(cfg["data_file"], build_vocab=False, char2idx_in=ds_all.char2idx_in, char2idx_out=ds_all.char2idx_out)
    train_ds.pairs = train_pairs
    val_ds = G2PDataset(cfg["data_file"], build_vocab=False, char2idx_in=ds_all.char2idx_in, char2idx_out=ds_all.char2idx_out)
    val_ds.pairs = val_pairs

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    input_dim = len(ds_all.char2idx_in)
    output_dim = len(ds_all.char2idx_out)
    char2idx_out = ds_all.char2idx_out
    idx2char_out = {i:c for c,i in char2idx_out.items()}
    # expose to beam search via module-level name used inside beam_search
    char2idx_out_global = char2idx_out

    print(f"Train size: {len(train_ds)}  Val size: {len(val_ds)}  Vocab_in: {input_dim}  Vocab_out: {output_dim}")

    encoder = Encoder(input_dim, cfg["emb_size"], cfg["hidden_size"], n_layers=cfg["num_layers"]).to(device)
    decoder = Decoder(output_dim, cfg["emb_size"], cfg["hidden_size"], enc_hid_dim=cfg["hidden_size"]*2).to(device)
    model = Seq2Seq(encoder, decoder, device, char2idx_out).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=ds_all.char2idx_out[PAD])

    best_val_cer = 1.0
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler() if (cfg["use_amp"] and device.type=="cuda") else None

    print("Starting training...")
    for epoch in range(1, cfg["epochs"]+1):
        start = time.time()
        frac = min(epoch / max(1, cfg["epochs"]), 1.0)
        tf_ratio = cfg["tf_start"] + frac * (cfg["tf_end"] - cfg["tf_start"])
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, tf_ratio, scaler=scaler)
        val_loss, val_cer = evaluate(model, val_loader, criterion, idx2char_out)
        scheduler.step(val_loss)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{cfg['epochs']}  Time: {elapsed:.1f}s  TrainLoss: {train_loss:.4f}  ValLoss: {val_loss:.4f}  ValCER: {val_cer:.4f}  TF: {tf_ratio:.3f}")

        if val_cer + 1e-12 < best_val_cer:
            best_val_cer = val_cer
            torch.save({
                "model_state": model.state_dict(),
                "char2idx_in": ds_all.char2idx_in,
                "char2idx_out": ds_all.char2idx_out,
                "config": {
                    "emb_size": cfg["emb_size"],
                    "hidden_size": cfg["hidden_size"],
                }
            }, cfg["checkpoint"])
            print(f"  Saved best model (CER {best_val_cer:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement (patience {patience_counter}/{cfg['patience']})")
            if patience_counter >= cfg["patience"]:
                print("Early stopping triggered.")
                break

    print("Training done. Best val CER:", best_val_cer)
    print("Best model saved to", cfg["checkpoint"])
