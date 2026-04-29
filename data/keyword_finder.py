"""
data/keyword_finder.py
Localização robusta de keywords nos tokens do prompt,
com múltiplas estratégias de fallback.
"""

from transformers import AutoTokenizer
from typing import Optional, List, Tuple, Dict
import unicodedata


# Mapeamento de keywords com variantes ortográficas.
# Cobre acentuação, capitalização e formas alternativas.
KEYWORD_VARIANTS = {
    # Avião
    "avião":    ["avião", "aviao", "Avião", "Aviao", "AVIÃO", " avião", "▁avião"],
    "airplane": ["airplane", "Airplane", "AIRPLANE", "aeroplane", "Aeroplane"],
    "Flugzeug": ["Flugzeug", "flugzeug", "FLUGZEUG", " Flugzeug"],
    "aereo":    ["aereo", "Aereo", "l'aereo", "l'aereo", " aereo"],

    # Telefone
    "telefone": ["telefone", "Telefone", "TELEFONE", " telefone"],
    "telephone":["telephone", "Telephone", "TELEPHONE", " telephone"],
    "Telefon":  ["Telefon", "telefon", "TELEFON", " Telefon"],
    "telefono": ["telefono", "Telefono", "il telefono", " telefono"],

    # Rádio
    "rádio":    ["rádio", "radio", "Rádio", "Radio", "RÁDIO", " rádio", "▁rádio"],
    "radio":    ["radio", "Radio", "RADIO", " radio", "▁radio"],
    "Radio":    ["Radio", "radio", "RADIO", " Radio"],
}


def find_keyword_position_robust(
    tokenizer: AutoTokenizer,
    input_ids: List[int],
    keyword: str,
    model_key: str
) -> Tuple[Optional[int], str]:
    """
    Encontra a posição do token da keyword com múltiplas
    estratégias de fallback.

    Estratégias (em ordem de prioridade):
        1. Match exato da keyword
        2. Match de variantes ortográficas
        3. Match por substring nos tokens decodificados
        4. Match após normalização Unicode (remove acentos)

    Args:
        tokenizer:  tokenizer do modelo
        input_ids:  lista de token ids do prompt
        keyword:    palavra-chave a encontrar (ex: "avião")
        model_key:  chave do modelo (para logging)

    Returns:
        Tupla (posição do último token da keyword, estratégia usada).
        Posição é None se não encontrada por nenhuma estratégia.
    """

    # ── Estratégia 1: Match exato ──
    pos = _exact_match(tokenizer, input_ids, keyword)
    if pos is not None:
        return pos, "exact"

    # ── Estratégia 2: Variantes ortográficas ──
    variants = KEYWORD_VARIANTS.get(keyword, [])
    for variant in variants:
        pos = _exact_match(tokenizer, input_ids, variant)
        if pos is not None:
            return pos, f"variant:{variant}"

    # ── Estratégia 3: Substring nos tokens decodificados ──
    pos = _substring_match(tokenizer, input_ids, keyword)
    if pos is not None:
        return pos, "substring"

    # ── Estratégia 4: Match após normalização Unicode ──
    keyword_normalized = _normalize_unicode(keyword)
    pos = _substring_match(
        tokenizer, input_ids, keyword_normalized, normalize=True
    )
    if pos is not None:
        return pos, "unicode_normalized"

    # Nenhuma estratégia funcionou
    print(f"  ⚠️  [{model_key}] Keyword '{keyword}' não encontrada")
    print(f"       Tokens: {_decode_all(tokenizer, input_ids[:20])}...")
    return None, "not_found"


def _exact_match(
    tokenizer: AutoTokenizer,
    input_ids: List[int],
    keyword: str
) -> Optional[int]:
    """
    Tokeniza a keyword e busca a subsequência exata nos input_ids.
    Busca da direita para a esquerda — queremos a última ocorrência
    (mais provável de ser a keyword no corpo da pergunta,
    não em tokens de template).
    Retorna a posição do último token da keyword.
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
    se algum contém a keyword como substring.

    Lida com casos onde a keyword é dividida em múltiplos tokens
    mas um deles contém a parte mais identificável.

    Também tenta janelas de 2-3 tokens consecutivos para
    keywords divididas como "avi" + "ão".

    Retorna a posição do token (ou último token da janela) que
    contém a keyword.
    """
    kw = _normalize_unicode(keyword) if normalize else keyword.lower()

    decoded = [
        tokenizer.decode([tid]).strip()
        for tid in input_ids
    ]

    # Busca simples — token único contém a keyword
    for i in range(len(decoded) - 1, -1, -1):
        token_text = (
            _normalize_unicode(decoded[i]) if normalize
            else decoded[i].lower()
        )
        if kw in token_text:
            return i

    # Busca em janelas de tokens consecutivos
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
    Ex: 'avião' → 'aviao', 'rádio' → 'radio'
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()


def _decode_all(
    tokenizer: AutoTokenizer,
    input_ids: List[int]
) -> List[str]:
    """Decodifica todos os tokens — usado para debug/logging."""
    return [tokenizer.decode([tid]) for tid in input_ids]


def verify_keyword_finding(
    tokenizers_dict: Dict,
    prompts_sample: Dict
):
    """
    Utilitário de verificação — testa a localização de keywords
    em todos os modelos antes de rodar o experimento completo.

    Imprime uma tabela mostrando:
    - Qual estratégia foi usada
    - Qual token foi encontrado
    - Se o resultado parece correto

    Args:
        tokenizers_dict: dict {model_key: tokenizer}
        prompts_sample:  dict de prompts (mesma estrutura de PROMPTS)
    """
    from config.experiment_config import ExperimentConfig
    from models.chat_template import apply_template

    config = ExperimentConfig()

    print("\n" + "=" * 60)
    print("VERIFICAÇÃO: KEYWORD FINDING")
    print("=" * 60)

    for model_key, tokenizer in tokenizers_dict.items():
        print(f"\n[{model_key.upper()}]")

        for fact in ["aviao", "telefone", "radio"]:
            for lang in ["pt", "en", "de", "it"]:

                keyword   = config.keyword_tokens[fact][lang]
                prompt    = prompts_sample[fact][lang]["F1"]

                formatted, _ = apply_template(tokenizer, model_key, prompt)
                input_ids = tokenizer.encode(
                    formatted, add_special_tokens=False
                )

                pos, strategy = find_keyword_position_robust(
                    tokenizer, input_ids, keyword, model_key
                )

                if pos is not None:
                    found_token = tokenizer.decode([input_ids[pos]])
                    status = "✅" if strategy == "exact" else "⚠️ "
                else:
                    found_token = "NOT FOUND"
                    status = "❌"

                print(
                    f"  {status} [{fact}][{lang}]  "
                    f"keyword='{keyword:12s}'  "
                    f"pos={str(pos):4s}  "
                    f"token='{found_token:12s}'  "
                    f"strategy={strategy}"
                )
