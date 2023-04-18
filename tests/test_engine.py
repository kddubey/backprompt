"""
Unit tests `backprompt.engine.Text` by comparing its outputs to those from plain model
calls.
TODO: expand
"""
from __future__ import annotations

import pytest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backprompt.engine import Text


@pytest.fixture(scope="module")
def atol():
    ## Reading through some transformers tests, it looks like 1e-3 is considered
    ## "close enough" for hidden states. See, e.g.,
    ## https://github.com/huggingface/transformers/blob/main/tests/models/gpt2/test_modeling_gpt2.py#L250
    return 1e-4


@pytest.fixture(scope="module")
def model_name():
    ## There are a lot of tiny models on https://huggingface.co/sshleifer which are
    ## useful for testing code. Weights can be random.
    return "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def model(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name)


@pytest.fixture(scope="module")
def tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        ## allow padding -> allow batching
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@pytest.fixture(scope="module")
def model_and_tokenizer(model, tokenizer):
    return model, tokenizer


def test_backward(model_and_tokenizer, atol):
    context = Text("a", model_and_tokenizer)
    request1 = Text(" b c", model_and_tokenizer)
    request2 = Text(" d e", model_and_tokenizer)

    cr1 = context + request1
    cr1.backward()
    cr1.model_repr[1].logits

    model, tokenizer = model_and_tokenizer
    with torch.no_grad():
        out1 = model(**tokenizer(cr1.string, return_tensors="pt"))
    assert torch.allclose(
        out1.logits[0, -2:], cr1.model_repr[1].logits[0, -2:], atol=atol
    )

    cr12 = cr1 + request2
    cr12.backward()

    cr12.model_repr[1].logits
    with torch.no_grad():
        out12 = model(**tokenizer(cr12.string, return_tensors="pt"))
    assert torch.allclose(
        out12.logits[0, -2:], cr12.model_repr[1].logits[0, -2:], atol=atol
    )
