"""
logit_lens/lens_training.py
Treino e avaliação do Tuned Lens (Fase 4), retomável.

Dados (config/tuned_lens.yaml):
    - uma fonte por idioma (FineWeb-2 / FineWeb, streaming); cada fonte
      é uma função que devolve um iterador novo de {"text", "id"}
    - held-out: sha256(id) mod held_out_mod == 0 nunca entra no treino
    - filtro: fora todo documento que cita nome de candidato (qualquer
      caso, qualquer idioma) ou palavra-chave de fato do idioma
    - empacotamento: [BOS] + doc + [EOS] + doc ..., cortado em seq_len
    - balanceamento: as sequências de cada passo alternam os idiomas em
      rodízio, então todo idioma recebe o mesmo número de tokens

Treino (receita do tuned-lens 0.2.0):
    - forward do modelo congelado sem gradiente
    - para cada camada 1..L−1 e cada pedaço de posições:
          KL(final ‖ lens) = Σ_v p_final · (log p_final − log p_lens)
      média sobre os tokens do passo, somada sobre as camadas
    - gradiente cortado em norma global max_grad_norm (1.0, como o
      pacote) antes de cada passo; a norma antes do corte é registrada
    - SGD Nesterov, decaimento linear sem warmup

Receita: os hiperparâmetros que mudam o resultado, o corpus e
RECIPE_VERSION (recipe / recipe_sha256). Vai no checkpoint, em
lenses/{modelo}/RECIPE e no config.json da lente; um checkpoint de outra
receita é recusado (RecipeMismatch), nunca retomado.

Retomada: checkpoint com passo, tradutores, otimizador, scheduler,
posição de cada fonte (documentos lidos), buffer de empacotamento e
rodízio. Retomar dá o mesmo resultado que não interromper.

Aceitação (acceptance): depois da avaliação held-out, a lente só é
aceita se não for pior que o Logit Lens (+ tolerância) na última camada
traduzida e na média da zona de decisão.
"""

import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F

from data.text_matching import compile_terms, normalize_text

Source = Callable[[], Iterator[Dict]]

# 1: sem corte do gradiente (lentes de 2026-10-29); 2: com max_grad_norm (2026-10-30)
RECIPE_VERSION = 2
# Chaves de training que não mudam o resultado (só memória / frequência de checkpoint)
NOT_IN_RECIPE = ("loss_chunk_tokens", "checkpoint_every")


class RecipeMismatch(RuntimeError):
    """Checkpoint ou lente de outra receita de treino."""


def recipe(tl_cfg: Dict) -> Dict:
    """O que decide o resultado do treino (sem o que só afeta memória)."""
    t = {k: v for k, v in tl_cfg["training"].items() if k not in NOT_IN_RECIPE}
    return {"recipe_version": RECIPE_VERSION, "training": t, "corpus": tl_cfg["corpus"],
            "held_out_mod": tl_cfg["held_out_mod"], "filter_fact_terms": tl_cfg["filter_fact_terms"]}


def recipe_sha256(tl_cfg: Dict) -> str:
    blob = json.dumps(recipe(tl_cfg), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════
# Dados
# ══════════════════════════════════════════════════════════════

def is_held_out(doc_id, mod: int) -> bool:
    h = hashlib.sha256(str(doc_id).encode("utf-8")).hexdigest()
    return int(h, 16) % mod == 0


def fact_terms(facts: Dict, lang: str) -> List[str]:
    """Nomes de todos os candidatos (todos os casos e idiomas) + palavras-chave do idioma."""
    terms = set()
    for f in facts.values():
        if lang in f.keyword:
            terms.add(f.keyword[lang])
        for c in f.candidates:
            for by_case in c.names.values():
                for forms in by_case.values():
                    terms.update(forms)
    return sorted(terms)


class DocStream:
    """
    Documentos de uma fonte, já separados (treino ou held-out) e
    filtrados. `consumed` conta documentos crus lidos da fonte, para
    retomar com islice.
    """

    def __init__(self, source: Source, split: str, held_out_mod: int,
                 pattern=None, text_key="text", id_key="id", skip: int = 0):
        assert split in ("train", "eval")
        self.split, self.mod, self.pattern = split, held_out_mod, pattern
        self.text_key, self.id_key = text_key, id_key
        self.consumed = skip
        self.stats = {"read": 0, "held_out": 0, "filtered": 0, "kept": 0}
        self._it = itertools.islice(source(), skip, None)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        while True:
            rec = next(self._it)
            self.consumed += 1
            self.stats["read"] += 1
            held = is_held_out(rec[self.id_key], self.mod)
            if held:
                self.stats["held_out"] += 1
            if held != (self.split == "eval"):
                continue
            text = rec[self.text_key]
            if self.pattern is not None and self.pattern.search(normalize_text(text)):
                self.stats["filtered"] += 1
                continue
            self.stats["kept"] += 1
            return text


class Packer:
    """[BOS] doc [EOS] doc ... em sequências de seq_len tokens."""

    def __init__(self, docs: Iterator[str], tokenizer, seq_len: int, buffer=None):
        self.docs, self.tok, self.seq_len = docs, tokenizer, seq_len
        self.buffer = list(buffer or [])
        self.bos = getattr(tokenizer, "bos_token_id", None)
        self.eos = tokenizer.eos_token_id

    def _encode(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return list(ids.tolist() if hasattr(ids, "tolist") else ids)

    def next_sequence(self) -> List[int]:
        # Primeira posição: BOS (o modelo espera); depois só o conteúdo
        body = self.seq_len - (1 if self.bos is not None else 0)
        while len(self.buffer) < body:
            self.buffer += self._encode(next(self.docs)) + [self.eos]
        seq, self.buffer = self.buffer[:body], self.buffer[body:]
        return ([self.bos] if self.bos is not None else []) + seq


class BalancedBatches:
    """
    Sequências de treino em rodízio entre idiomas. Estado serializável
    (posição de cada fonte, buffers, ponteiro do rodízio).
    """

    def __init__(self, sources: Dict[str, Source], tokenizer, seq_len: int,
                 held_out_mod: int, patterns: Dict[str, object], keys: Dict[str, Dict],
                 state: Optional[Dict] = None, split: str = "train"):
        self.langs = list(sources)
        state = state or {}
        self.pointer = state.get("pointer", 0)
        self.streams, self.packers = {}, {}
        for l in self.langs:
            s = state.get("langs", {}).get(l, {})
            self.streams[l] = DocStream(sources[l], split, held_out_mod, patterns.get(l),
                                        keys[l].get("text", "text"), keys[l].get("id", "id"),
                                        skip=s.get("consumed", 0))
            self.packers[l] = Packer(self.streams[l], tokenizer, seq_len, s.get("buffer"))

    def next(self, n: int) -> List[tuple]:
        out = []
        for _ in range(n):
            lang = self.langs[self.pointer % len(self.langs)]
            self.pointer += 1
            out.append((lang, self.packers[lang].next_sequence()))
        return out

    def state(self) -> Dict:
        return {"pointer": self.pointer,
                "langs": {l: {"consumed": self.streams[l].consumed,
                              "buffer": list(self.packers[l].buffer)} for l in self.langs}}

    def stats(self) -> Dict:
        return {l: dict(self.streams[l].stats) for l in self.langs}


# ══════════════════════════════════════════════════════════════
# Perda
# ══════════════════════════════════════════════════════════════

def _forward_hidden(model, ids: List[int]):
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True)
    return out.hidden_states, out.logits[0]


def sequence_kl(lens, model, ids: List[int], chunk: int, backward_scale: Optional[float] = None,
                translate: bool = True) -> torch.Tensor:
    """
    KL(final ‖ lens) somada sobre as posições, por camada 1..L−1
    (tensor [L−1], sem gradiente). Com backward_scale, faz backward de
    cada pedaço com a perda multiplicada por esse fator (treino).
    translate=False mede o Logit Lens (sem tradutor), para comparação.
    """
    hs, final = _forward_hidden(model, ids)
    n_layers  = len(hs) - 1
    T = len(ids)
    sums = torch.zeros(n_layers - 1)
    for k in range(1, n_layers):
        for a in range(0, T, chunk):
            b = min(a + chunk, T)
            target = F.log_softmax(final[a:b].float(), dim=-1)
            h = hs[k][0, a:b]
            if translate:
                logits = lens.layer_logits_grad(k, h)
            else:
                with torch.no_grad():
                    logits = lens.head(h, already_normed=False)
            logq = F.log_softmax(logits, dim=-1)
            kl = (target.exp() * (target - logq.to(target.device))).sum()
            if backward_scale is not None:
                (kl * backward_scale).backward()
            sums[k - 1] += kl.detach().float().cpu()
    return sums


# ══════════════════════════════════════════════════════════════
# Treino
# ══════════════════════════════════════════════════════════════

def make_optimizer(lens, tcfg: Dict, num_steps: int):
    from transformers import get_linear_schedule_with_warmup
    opt = torch.optim.SGD(lens.parameters(), lr=tcfg["lr"], momentum=tcfg["momentum"],
                          nesterov=tcfg["nesterov"], weight_decay=tcfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, 0, num_steps)
    return opt, sched


def _save_checkpoint(path: Path, ckpt: Dict):
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def train(lens, model, tokenizer, sources: Dict[str, Source], tl_cfg: Dict, facts: Dict,
          out_dir, max_steps: Optional[int] = None, log=print) -> Dict:
    """
    Treina (ou retoma) os tradutores. Salva checkpoint a cada
    checkpoint_every passos e no fim. `max_steps` para antes do fim
    (testes / sessões curtas); o agendamento usa sempre num_steps.

    Returns:
        dict com step, losses (média por passo) e stats do corpus
    """
    tcfg   = tl_cfg["training"]
    out    = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_p = out / "checkpoint.pt"
    steps  = tcfg["num_steps"]
    n_seq, rem = divmod(tcfg["tokens_per_step"], tcfg["seq_len"])
    if rem:
        raise ValueError("tokens_per_step precisa ser múltiplo de seq_len")

    model.requires_grad_(False)          # só os tradutores aprendem
    torch.manual_seed(tcfg["seed"])
    max_norm = float(tcfg["max_grad_norm"])
    sha = recipe_sha256(tl_cfg)
    opt, sched = make_optimizer(lens, tcfg, steps)
    state = {"step": 0, "losses": [], "grad_norms": [], "data": None}
    if ckpt_p.exists():
        ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        if ck.get("recipe_sha256") != sha:
            raise RecipeMismatch(
                f"{ckpt_p} é de outra receita de treino ({ck.get('recipe_sha256') or 'sem hash: receita 1'}"
                f" ≠ {sha[:12]}). Não retomo: mova {out} para lenses_stale/ (train_tuned_lens.py faz isso).")
        lens.translators.load_state_dict(ck["translators"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["rng"])
        state = {"step": ck["step"], "losses": ck["losses"], "grad_norms": ck.get("grad_norms", []),
                 "data": ck["data"]}
        log(f"  retomando do passo {state['step']}/{steps}")

    patterns = {l: compile_terms(fact_terms(facts, l)) if tl_cfg["filter_fact_terms"] else None
                for l in sources}
    keys = {l: tl_cfg["corpus"].get(l, {}) for l in sources}
    batches = BalancedBatches(sources, tokenizer, tcfg["seq_len"], tl_cfg["held_out_mod"],
                              patterns, keys, state=state["data"])
    stop = steps if max_steps is None else min(steps, max_steps)

    while state["step"] < stop:
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        seqs  = batches.next(n_seq)
        scale = 1.0 / (n_seq * tcfg["seq_len"])
        total = torch.zeros(())
        for _, ids in seqs:
            total += sequence_kl(lens, model, ids, tcfg["loss_chunk_tokens"],
                                 backward_scale=scale).sum() * scale
        # Corte do gradiente como no pacote (norma global de todos os tradutores)
        gnorm = float(torch.nn.utils.clip_grad_norm_(lens.parameters(), max_norm))
        state["grad_norms"].append(gnorm)
        opt.step()
        sched.step()
        state["step"] += 1
        state["losses"].append(float(total))
        log(f"  passo {state['step']}/{steps}  KL (soma das camadas) = {float(total):.4f}  "
            f"|grad| antes do corte = {gnorm:.2f}  ({time.time() - t0:.1f}s)")
        if state["step"] % tcfg["checkpoint_every"] == 0 or state["step"] == stop:
            _save_checkpoint(ckpt_p, {
                "step": state["step"], "losses": state["losses"], "grad_norms": state["grad_norms"],
                "recipe_sha256": sha,
                "translators": lens.translators.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "rng": torch.get_rng_state(), "data": batches.state()})
    return {"step": state["step"], "num_steps": steps, "losses": state["losses"],
            "grad_norms_preclip": state["grad_norms"], "max_grad_norm": max_norm,
            "recipe_sha256": sha, "recipe_version": RECIPE_VERSION,
            "corpus_stats": batches.stats(), "done": state["step"] >= steps}


# ══════════════════════════════════════════════════════════════
# Avaliação held-out
# ══════════════════════════════════════════════════════════════

def evaluate(lens, model, tokenizer, sources: Dict[str, Source], tl_cfg: Dict,
             facts: Dict) -> List[Dict]:
    """
    KL(final ‖ lens) média por token, por idioma e camada, no held-out,
    para o Tuned Lens e para o Logit Lens (mesmas sequências).
    """
    tcfg = tl_cfg["training"]
    n_seq = max(1, tl_cfg["eval"]["tokens_per_language"] // tcfg["seq_len"])
    rows = []
    for lang, src in sources.items():
        pat = compile_terms(fact_terms(facts, lang)) if tl_cfg["filter_fact_terms"] else None
        keys = tl_cfg["corpus"].get(lang, {})
        docs = DocStream(src, "eval", tl_cfg["held_out_mod"], pat,
                         keys.get("text", "text"), keys.get("id", "id"))
        packer = Packer(docs, tokenizer, tcfg["seq_len"])
        tuned = logit = None
        n_tok = 0
        for _ in range(n_seq):
            try:
                ids = packer.next_sequence()
            except StopIteration:      # held-out esgotado: avalia o que houver
                break
            with torch.no_grad():
                t = sequence_kl(lens, model, ids, tcfg["loss_chunk_tokens"])
            l = sequence_kl(lens, model, ids, tcfg["loss_chunk_tokens"], translate=False)
            tuned = t if tuned is None else tuned + t
            logit = l if logit is None else logit + l
            n_tok += len(ids)
        if tuned is None:
            continue
        for k in range(len(tuned)):
            rows.append({"lang": lang, "layer": k + 1, "n_tokens": n_tok,
                         "kl_tuned": float(tuned[k] / n_tok), "kl_logit": float(logit[k] / n_tok)})
    return rows


# ══════════════════════════════════════════════════════════════
# Aceitação
# ══════════════════════════════════════════════════════════════

def acceptance(rows: List[Dict], n_layers: int, decision_zone: float, tol: float) -> Dict:
    """
    Teste de aceitação sobre a avaliação held-out (média dos idiomas por
    camada). Reprova se o Tuned Lens for pior que o Logit Lens + tol na
    última camada traduzida (L−1) ou na média da zona de decisão
    (camadas k > L·(1 − zone), as traduzidas). `failing_layers`: camadas
    da zona em que o Tuned Lens passa o Logit Lens + tol (informativo).
    """
    import pandas as pd
    g = pd.DataFrame(rows).groupby("layer")[["kl_tuned", "kl_logit"]].mean()
    last = n_layers - 1
    zone = g[g.index > n_layers * (1 - decision_zone) + 1e-9]
    lt, ll = float(g.loc[last, "kl_tuned"]), float(g.loc[last, "kl_logit"])
    # Zona sem camada traduzida (só em modelos minúsculos): critério da zona não se aplica
    zt = float(zone["kl_tuned"].mean()) if len(zone) else None
    zl = float(zone["kl_logit"].mean()) if len(zone) else None
    failing = [{"layer": int(k), "kl_tuned": round(float(r.kl_tuned), 4), "kl_logit": round(float(r.kl_logit), 4)}
               for k, r in zone.iterrows() if r.kl_tuned > r.kl_logit + tol]
    reasons = []
    if not lt <= ll + tol:                              # NaN reprova
        reasons.append(f"última camada traduzida ({last}): tuned {lt:.3f} > logit {ll:.3f} + {tol}")
    if zt is not None and not zt <= zl + tol:
        reasons.append(f"média da zona de decisão: tuned {zt:.3f} > logit {zl:.3f} + {tol}")
    rnd = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {"passed": not reasons, "reasons": reasons, "tolerance_nats": tol,
            "last_layer": last, "last_kl_tuned": rnd(lt), "last_kl_logit": rnd(ll),
            "zone_layers": [int(k) for k in zone.index], "zone_kl_tuned": rnd(zt),
            "zone_kl_logit": rnd(zl), "failing_layers": failing}


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
