"""
scripts/make_anonymous_copy.py
Cópia anonimizada do repositório para revisão cega.

    python scripts/make_anonymous_copy.py                       # → ../<repo>-anonymous
    python scripts/make_anonymous_copy.py --out /tmp/anon --force
    python scripts/make_anonymous_copy.py --scrub "Outro Nome" --scrub outro@mail

O que entra: os arquivos versionados (git ls-files), sem o histórico git
(.git não é copiado) e sem segredos (.env, *.env, arquivos de token).
Resultados, figuras, lentes e caches não são versionados e ficam de fora.

O que é trocado, em todo arquivo de texto (palavras inteiras, sem caixa):
    nome e e-mail dos autores dos commits (git log), do git config
    (user.name, user.email), o usuário do GitHub do remote origin e os
    termos de --scrub                                   → "Anonymous"
    qualquer e-mail                                     → anonymous@example.com
    tokens do Hugging Face (hf_...)                     → hf_REDACTED
    caminhos de home (/Users/<x>/, /home/<x>/)          → ~/
    URLs do repositório (github.com/<usuário>/<repo>)   → anonymous repository

Depois da cópia, tudo é varrido de novo (texto e binário), com a mesma
regra de palavra inteira da troca (um termo dentro de outra palavra, como
um primeiro nome dentro de "Leonardo", não conta); se sobrar qualquer
termo, o script falha (código 1) e lista os arquivos. Nenhum
relatório com os termos originais é gravado dentro da cópia.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET_NAMES = re.compile(r"(^|/)(\.env(\..*)?|.*\.env|.*token.*\.txt|\.netrc|credentials.*)$", re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{20,}")
HOME = re.compile(r"/(Users|home)/[^/\s\"']+/")
ANON = "Anonymous"


def _git(*args, cwd=ROOT) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def identity_terms(extra=(), src=ROOT, from_config=True) -> list:
    """
    Nomes e e-mails dos autores dos commits, usuário do GitHub do remote,
    o git config local (from_config) e termos extras. Sem o git config,
    o resultado só depende do repositório (igual em qualquer máquina).
    """
    terms = []
    for line in _git("log", "--format=%an%n%ae", cwd=src).splitlines():
        terms.append(line)
    if from_config:
        terms += [_git("config", "user.name", cwd=src), _git("config", "user.email", cwd=src)]
    m = re.search(r"github\.com[:/]([^/]+)/", _git("remote", "get-url", "origin", cwd=src))
    if m:
        terms.append(m.group(1))
    terms += list(extra)
    out = []
    for t in terms:
        t = t.strip()
        if t and t not in out:
            out.append(t)
            # partes do nome (primeiro nome e sobrenome) também
            if " " in t and "@" not in t:
                out += [p for p in t.split() if len(p) > 2 and p not in out]
    return out


def scrub_text(text: str, terms: list, repo_slug: str = None) -> tuple:
    """(texto anonimizado, número de trocas)."""
    n = 0

    def sub(pattern, repl, s, flags=0):
        nonlocal n
        s2, k = re.subn(pattern, repl, s, flags=flags)
        n += k
        return s2

    if repo_slug:
        text = sub(r"https?://github\.com/" + re.escape(repo_slug) + r"(\.git)?", "anonymous repository", text)
        text = sub(re.escape(repo_slug), "anonymous/repository", text)
    text = sub(HF_TOKEN.pattern, "hf_REDACTED", text)
    text = sub(EMAIL.pattern, "anonymous@example.com", text)
    text = sub(HOME.pattern, "~/", text)
    for t in sorted(terms, key=len, reverse=True):      # nome completo antes das partes
        if "@" in t:
            continue
        text = sub(r"(?<!\w)" + re.escape(t) + r"(?!\w)", ANON, text, flags=re.I)
    text = text.replace(f"{ANON} {ANON}", ANON)
    return text, n


def _word(term: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.I)


def leftovers(data: bytes, terms: list) -> list:
    """Termos que sobraram como palavra inteira (mesma regra de scrub_text)."""
    text = data.decode("utf-8", "ignore")
    hits = [t for t in terms if _word(t).search(text)]
    if HF_TOKEN.search(text):
        hits.append("hf_ token")
    return hits


def make_copy(out: Path, terms: list, force=False, src=ROOT, log=print) -> dict:
    files = _git("ls-files", cwd=src).splitlines()
    if not files:
        raise RuntimeError(f"nenhum arquivo versionado em {src}")
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} já existe (use --force)")
        shutil.rmtree(out)
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", _git("remote", "get-url", "origin", cwd=src))
    slug = m.group(1) if m else None
    stats = {"copied": 0, "skipped_secret": [], "missing": 0, "replacements": 0, "problems": []}
    for rel in files:
        if SECRET_NAMES.search(rel):
            stats["skipped_secret"].append(rel)
            continue
        p = src / rel
        if not p.is_file():
            stats["missing"] += 1
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = p.read_bytes()
        if b"\0" in data[:8192]:                       # binário: copia e só verifica
            dst.write_bytes(data)
        else:
            text, n = scrub_text(data.decode("utf-8", "replace"), terms, slug)
            dst.write_text(text, encoding="utf-8")
            stats["replacements"] += n
        stats["copied"] += 1
    for f in out.rglob("*"):
        if f.is_file():
            hits = leftovers(f.read_bytes(), terms + ([slug] if slug else []))
            if hits:
                stats["problems"].append(str(f.relative_to(out)))
    return stats


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT.parent / f"{ROOT.name}-anonymous"))
    p.add_argument("--scrub", action="append", default=[], help="termo extra a remover (repetível)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    terms = identity_terms(args.scrub)
    stats = make_copy(Path(args.out), terms, force=args.force)
    print(f"Cópia anônima em {args.out}: {stats['copied']} arquivos, "
          f"{stats['replacements']} trocas, {len(stats['skipped_secret'])} arquivo(s) secreto(s) "
          f"fora ({', '.join(stats['skipped_secret']) or '—'}), {len(terms)} termos de identidade.")
    if stats["problems"]:
        print("❌ termos de identidade ainda presentes em:")
        for f in stats["problems"]:
            print(f"   {f}")
        return 1
    print("✅ nenhum termo de identidade ou token restante")
    return 0


if __name__ == "__main__":
    sys.exit(main())
