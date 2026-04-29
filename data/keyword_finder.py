from transformers import AutoTokenizer
from typing import Optional, List, Tuple, Dict
import unicodedata


# Mapeamento de keywords com variantes ortográficas
# Cobre acentuação, capitalização e formas alternativas
KEYWORD_VARIANTS = {
    "avião":    ["avião", "aviao", "Avião", "Aviao", "AVIÃO"],
    "airplane": ["airplane", "Airplane", "AIRPLANE", "aeroplane"],
    "Flugzeug": ["Flugzeug", "flugzeug", "FLUGZEUG"],
    "aereo":    ["aereo", "Aereo", "l'aereo", "aereo"],
    
    "telefone": ["telefone", "Telefone", "TELEFONE"],
    "telephone":["telephone", "Telephone", "TELEPHONE"],
    "Telefon":  ["Telefon", "telefon", "TELEFON"],
    "telefono": ["telefono", "Telefono", "il telefono"],
    
    "rádio":    ["rádio", "radio", "Rádio", "Radio", "RÁDIO"],
    "radio":    ["radio", "Radio", "RADIO"],
    "Radio":    ["Radio", "radio", "RADIO"],
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
        keyword:    palavra-chave a encontrar
        model_key:  chave do modelo (para logging)
    
    Returns:
        Tupla (posição do último token da keyword, estratégia usada)
        Posição é None se não encontrada por nenhuma estratégia
    """
    
    # === Estratégia 1: Match exato ===
    pos = _exact_match(tokenizer, input_ids, keyword)
    if pos is not None:
        return pos, "exact"
    
    # === Estratégia 2: Variantes ortográficas ===
    variants = KEYWORD_VARIANTS.get(keyword, [])
    for variant in variants:
        pos = _exact_match(tokenizer, input_ids, variant)
        if pos is not None:
            return pos, f"variant:{variant}"
    
    # === Estratégia 3: Substring nos tokens decodificados ===
    pos = _substring_match(tokenizer, input_ids, keyword)
    if pos is not None:
        return pos, "substring"
    
    # === Estratégia 4: Match após normalização Unicode ===
    keyword_normalized = _normalize_unicode(keyword)
    pos = _substring_match(
        tokenizer, input_ids, keyword_normalized,
        normalize=True
    )
    if pos is not None:
        return pos, "unicode_normalized"
    
    # Nenhuma estratégia funcionou
    print(f"  ⚠️  [{model_key}] Keyword '{keyword}' não encontrada")
    print(f"       Tokens do prompt: {_decode_all(tokenizer, input_ids)}")
    return None, "not_found"


def _exact_match(
    tokenizer: AutoTokenizer,
    input_ids: List[int],
    keyword: str
) -> Optional[int]:
    """
    Tokeniza a keyword e busca a subsequência exata nos input_ids.
    Retorna a posição do último token da keyword.
    """
    keyword_ids = tokenizer.encode(
        keyword,
        add_special_tokens=False
    )
    
    kw_len = len(keyword_ids)
    
    # Busca da direita para a esquerda — queremos a última ocorrência
    # (mais provável de ser a keyword no corpo da pergunta,
    #  não em tokens de template)
    for i in range(len(input_ids) - kw_len, -1, -1):
        if input_ids[i:i + kw_len] == keyword_ids:
            return i + kw_len - 1  # posição do último token
    
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
    
    Retorna a posição do token que contém a keyword.
    """
    kw = _normalize_unicode(keyword) if normalize else keyword.lower()
    
    # Decodificar tokens individualmente
    decoded = [
        tokenizer.decode([tid]).strip()
        for tid in input_ids
    ]
    
    # Buscar substring da direita para esquerda
    for i in range(len(decoded) - 1, -1, -1):
        token_text = (
            _normalize_unicode(decoded[i]) if normalize 
            else decoded[i].lower()
        )
        if kw in token_text:
            return i
    
    # Tentar match em janelas de 2-3 tokens consecutivos
    # (para keywords divididas como "avi" + "ão")
    for window_size in [2, 3]:
        for i in range(len(decoded) - window_size + 1):
            window_text = "".join(decoded[i:i + window_size]).lower()
            if normalize:
                window_text = _normalize_unicode(window_text)
            if kw in window_text:
                # Retornar o último token da janela
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
    """Decodifica todos os tokens para debug."""
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
    """
    print("\n" + "="*60)
    print("VERIFICAÇÃO DE KEYWORD FINDING")
    print("="*60)
    
    from config.experiment_config import ExperimentConfig
    config = ExperimentConfig()
    
    for model_key, tokenizer in tokenizers_dict.items():
        print(f"\n[{model_key.upper()}]")
        
        for fact in ["aviao", "telefone", "radio"]:
            for lang in ["pt", "en", "de", "it"]:
                
                keyword = config.keyword_tokens[fact][lang]
                prompt = prompts_sample[fact][lang]["F1"]
                
                # Aplicar chat template
                from models.chat_template import apply_template
                formatted_prompt, _ = apply_template(
                    tokenizer, model_key, prompt
                )
                
                input_ids = tokenizer.encode(
                    formatted_prompt,
                    add_special_tokens=False
                )
                
                pos, strategy = find_keyword_position_robust(
                    tokenizer, input_ids, keyword, model_key
                )
                
                if pos is not None:
                    found_token = tokenizer.decode([input_ids[pos]])
                    status = "✅"
                else:
                    found_token = "NOT FOUND"
                    status = "❌"
                
                print(
                    f"  {status} [{fact}][{lang}] "
                    f"keyword='{keyword}' → "
                    f"pos={pos}, token='{found_token}', "
                    f"strategy={strategy}"
                )