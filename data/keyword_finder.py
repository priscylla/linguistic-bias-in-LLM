"""
data/keyword_finder.py
Localizacao robusta de keywords nos tokens do prompt,
com multiplas estrategias de fallback.
"""

from transformers import AutoTokenizer
from typing import Optional, List, Tuple, Dict
import unicodedata


# Mapeamento de keywords com variantes ortograficas.
# Inclui:
#   - versoes com e sem acento
#   - versoes com espaco prefixado (SentencePiece/BPE tokenizam assim)
#   - sufixos comuns quando a keyword e quebrada em subtokens
#   - prefixo especial de SentencePiece (unicode underscore)
KEYWORD_VARIANTS = {
    # ── Aviao ──
    "aviao": [
        "aviao", "Aviao", "AVIAO",
        " aviao", " Aviao",          # espaco prefixado (LLaMA, Mistral)
        "aviao",                      # sem acento
        "\u2581aviao",               # underscore SentencePiece
        "iao",                        # sufixo quando quebrado em av+iao
        "viao",                       # sufixo alternativo
    ],
    "airplane": [
        "airplane", "Airplane", "AIRPLANE",
        " airplane", " Airplane",     # espaco prefixado
        "aeroplane", " aeroplane",
        "\u2581airplane",
        "plane", " plane",            # sufixo quando quebrado
    ],
    "Flugzeug": [
        "Flugzeug", "flugzeug", "FLUGZEUG",
        " Flugzeug", " flugzeug",     # espaco prefixado
        "\u2581Flugzeug",
        "zeug", "Zeug",               # sufixo quando quebrado em Flug+zeug
        "lugzeug",                    # sufixo alternativo
    ],
    "aereo": [
        "aereo", "Aereo",
        " aereo", " Aereo",           # espaco prefixado
        "l'aereo", "l\u2019aereo",   # forma italiana com artigo
        "\u2581aereo",
        "ereo",                       # sufixo quando quebrado em a+ereo
        "areo",                       # variante ortografica
    ],

    # ── Telefone ──
    "telefone": [
        "telefone", "Telefone", "TELEFONE",
        " telefone", " Telefone",     # espaco prefixado
        "\u2581telefone",
        "lefone", "fone",             # sufixos quando quebrado
    ],
    "telephone": [
        "telephone", "Telephone", "TELEPHONE",
        " telephone", " Telephone",   # espaco prefixado
        "\u2581telephone",
        "lephone", "phone",           # sufixos quando quebrado
    ],
    "Telefon": [
        "Telefon", "telefon", "TELEFON",
        " Telefon", " telefon",       # espaco prefixado
        "\u2581Telefon",
        "lefon", "fon",               # sufixos quando quebrado
    ],
    "telefono": [
        "telefono", "Telefono",
        " telefono", " Telefono",     # espaco prefixado
        "il telefono",
        "\u2581telefono",
        "lefono", "fono",             # sufixos quando quebrado
    ],
}


def find_keyword_position_robust(
    tokenizer: AutoTokenizer,
    input_ids: List[int],
    keyword: str,
    model_key: str
) -> Tuple[Optional[int], str]:
    """
    Encontra a posicao do token da keyword com multiplas
    estrategias de fallback.

    Estrategias (em ordem de prioridade):
        1. Match exato da keyword
        2. Match de variantes do KEYWORD_VARIANTS
        3. Match por substring nos tokens decodificados
        4. Match apos normalizacao Unicode (remove acentos)

    Args:
        tokenizer:  tokenizer do modelo
        input_ids:  lista de token ids do prompt
        keyword:    palavra-chave a encontrar (ex: "aviao")
        model_key:  chave do modelo (para logging)

    Returns:
        Tupla (posicao do ultimo token da keyword, estrategia usada).
        Posicao e None se nao encontrada por nenhuma estrategia.
    """

    # ── Estrategia 1: Match exato ──
    pos = _exact_match(tokenizer, input_ids, keyword)
    if pos is not None:
        return pos, "exact"

    # ── Estrategia 2: Variantes do dicionario ──
    variants = KEYWORD_VARIANTS.get(keyword, [])
    for variant in variants:
        pos = _exact_match(tokenizer, input_ids, variant)
        if pos is not None:
            return pos, f"variant:{variant}"

    # ── Estrategia 3: Substring nos tokens decodificados ──
    pos = _substring_match(tokenizer, input_ids, keyword)
    if pos is not None:
        return pos, "substring"

    # ── Estrategia 4: Normalizacao Unicode (remove acentos) ──
    keyword_normalized = _normalize_unicode(keyword)
    pos = _substring_match(
        tokenizer, input_ids, keyword_normalized, normalize=True
    )
    if pos is not None:
        return pos, "unicode_normalized"

    print(f"  [AVISO] [{model_key}] Keyword '{keyword}' nao encontrada")
    print(f"  Tokens do prompt: {_decode_all(tokenizer, input_ids[:30])}...")
    return None, "not_found"


def _exact_match(
    tokenizer: AutoTokenizer,
    input_ids: List[int],
    keyword: str
) -> Optional[int]:
    """
    Tokeniza a keyword e busca a subsequencia exata nos input_ids.
    Busca da direita para esquerda — ultima ocorrencia e a mais
    provavel de ser a keyword no corpo da pergunta (nao no template).
    Retorna a posicao do ultimo token da keyword.
    """
    keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
    kw_len = len(keyword_ids)

    for i in range(len(input_ids) - kw_len, -1, -1):
        if input_ids[i:i + kw_len] == keyword_ids:
            return i + kw_len - 1

    return None


def _substring_match(
    tokenizer: AutoTokenizer,
    input_ids: List[int],
    keyword: str,
    normalize: bool = False
) -> Optional[int]:
    """
    Decodifica cada token individualmente e verifica
    se algum contem a keyword como substring.

    Tambem tenta janelas de 2-3 tokens consecutivos para
    keywords quebradas como "av" + "iao" -> "aviao".

    Retorna a posicao do token (ou ultimo token da janela)
    que contem a keyword.
    """
    kw = _normalize_unicode(keyword) if normalize else keyword.lower()

    decoded = [
        tokenizer.decode([tid]).strip()
        for tid in input_ids
    ]

    # Busca em token unico — da direita para esquerda
    for i in range(len(decoded) - 1, -1, -1):
        token_text = (
            _normalize_unicode(decoded[i]) if normalize
            else decoded[i].lower()
        )
        if kw in token_text:
            return i

    # Busca em janelas de 2-3 tokens consecutivos
    for window_size in [2, 3]:
        for i in range(len(decoded) - window_size + 1):
            window_text = "".join(decoded[i:i + window_size]).lower()
            if normalize:
                window_text = _normalize_unicode(window_text)
            if kw in window_text:
                return i + window_size - 1

    return None


def _normalize_unicode(text: str) -> str:
    """
    Remove acentos e normaliza para ASCII.
    Ex: 'aviao' -> 'aviao'
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()


def _decode_all(
    tokenizer: AutoTokenizer,
    input_ids: List[int]
) -> List[str]:
    """Decodifica todos os tokens -- usado para debug/logging."""
    return [tokenizer.decode([tid]) for tid in input_ids]


def verify_keyword_finding(
    tokenizers_dict: Dict,
    prompts_sample: Dict
):
    """
    Utilitario de verificacao -- testa a localizacao de keywords
    em todos os modelos antes de rodar o experimento completo.
    """
    from config.experiment_config import ExperimentConfig
    from models.chat_template import apply_template

    config = ExperimentConfig()

    print("\n" + "=" * 60)
    print("VERIFICACAO: KEYWORD FINDING")
    print("=" * 60)

    for model_key, tokenizer in tokenizers_dict.items():
        print(f"\n[{model_key.upper()}]")

        for fact in ["aviao", "telefone"]:
            for lang in ["pt", "en", "de", "it"]:

                keyword = config.keyword_tokens[fact][lang]
                prompt  = prompts_sample[fact][lang]["F1"]

                formatted, _ = apply_template(tokenizer, model_key, prompt)
                input_ids = tokenizer.encode(
                    formatted, add_special_tokens=False
                )

                pos, strategy = find_keyword_position_robust(
                    tokenizer, input_ids, keyword, model_key
                )

                if pos is not None:
                    found_token = tokenizer.decode([input_ids[pos]])
                    status = "OK " if strategy == "exact" else "OK*"
                else:
                    found_token = "NOT FOUND"
                    status = "ERR"

                print(
                    f"  {status} [{fact}][{lang}]  "
                    f"kw='{keyword:12s}'  "
                    f"pos={str(pos):4s}  "
                    f"token='{found_token:12s}'  "
                    f"{strategy}"
                )