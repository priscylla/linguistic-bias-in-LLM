"""
tests/conftest.py
Modelos minúsculos e aleatórios, construídos a partir de configs,
para testar o Logit Lens na CPU sem baixar nada.
"""

import sys
from pathlib import Path

import pytest

# Raiz do projeto no path para importar logit_lens/, models/, config/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.tiny import (  # noqa: E402,F401
    ARCHS, VOCAB, FakeTokenizer, WordTokenizer, _Batch, build_tiny_model,
)


@pytest.fixture(params=list(ARCHS))
def arch(request):
    return request.param


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def word_tokenizer():
    return WordTokenizer(splits={" Meucci": [" Me", "ucci"]})
