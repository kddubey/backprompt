from __future__ import annotations
from typing import Union

import torch
from transformers import BatchEncoding, GPT2Model, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


def _call_model_given_past(
    model: GPT2Model,
    tokenizer: GPT2Tokenizer,
    encoding_past: Union[BatchEncoding, None],
    out_past: Union[CausalLMOutputWithCrossAttentions, None],
    text: str,
) -> tuple[BatchEncoding, CausalLMOutputWithCrossAttentions]:
    if not tokenizer.padding_side == "right":
        raise ValueError(
            "Gotta use right padding to ensure position IDs are correct. "
            "Run tokenizer.padding_side = 'right' if sensible."
        )
    encoding: BatchEncoding = tokenizer([text], return_tensors="pt", padding=True).to(
        model.device
    )
    if encoding_past is None and out_past is None:  ## it's the first piece of text
        with torch.no_grad():
            out: CausalLMOutputWithCrossAttentions = model(**encoding)
        return encoding, out
    if encoding_past is None or out_past is None:
        raise ValueError(
            "encoding_past and out_past must both be None, or both not None."
        )
    ## Set position_ids to what they were had we fed the past text + text together w/
    ## right-padding (right b/c GPT-2 uses absolute position ids).
    ## This will be fixed in a future release. See:
    ## https://github.com/huggingface/transformers/issues/18104#issuecomment-1465629955
    _num_completion_tokens = encoding.input_ids.shape[1]
    position_ids = (
        torch.arange(_num_completion_tokens, device=model.device)
        + encoding_past.attention_mask.sum(dim=1)[:, None]
    )
    ## Need attention_mask to include the past since it prolly has padding
    attention_mask = torch.cat(
        (encoding_past.attention_mask, encoding.attention_mask), dim=1
    )
    ## Everything should now be aligned 🤞 🙏
    with torch.no_grad():
        out = model(
            input_ids=encoding.input_ids,
            attention_mask=attention_mask,
            past_key_values=out_past.past_key_values,
            position_ids=position_ids,
        )
    ## Concatenate the encodings for future model calls
    encoding = BatchEncoding(
        {key: torch.cat((encoding_past[key], encoding[key]), dim=1) for key in encoding}
    )
    return encoding, out


class Text:
    """
    Stores a single piece of text and its past keys and values.
    """

    def __init__(
        self,
        string: str,
        model_and_tokenizer: tuple[GPT2Model, GPT2Tokenizer],
        _prev=(),
    ):
        self.string = string
        self.model_and_tokenizer = model_and_tokenizer
        self.model_repr: tuple[BatchEncoding, CausalLMOutputWithCrossAttentions] = None
        ## internal variables used for Text graph construction
        self._forward = lambda: None  ## TODO: replace w/ partial'd model call?
        self._prev = _prev

    def __add__(self, other):
        other = other if isinstance(other, Text) else Text(other)
        out = Text(
            self.string + other.string, self.model_and_tokenizer, _prev=(self, other)
        )

        def _forward():
            if out.model_repr is None:
                out.model_repr = _call_model_given_past(
                    *self.model_and_tokenizer,
                    *self.model_repr,
                    other.string,
                )

        out._forward = _forward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo: list[Text] = []
        visited = set()

        def build_topo(text: Text):
            if text not in visited:
                visited.add(text)
                for child in text._prev:
                    build_topo(child)
                topo.append(text)

        build_topo(self)

        ## Initialize the root
        if topo[0].model_repr is None:
            topo[0].model_repr = _call_model_given_past(
                *self.model_and_tokenizer,
                encoding_past=None,
                out_past=None,
                text=topo[0].string,
            )
        for text in topo:
            text._forward()

    def __repr__(self) -> str:
        ## middle-truncate self.string if it's too long (co-written by ChatGPT)
        max_length = 20
        joiner = " ... "
        if len(self.string) <= max_length:
            string_shown = self.string
        else:
            truncate_len = max_length - len(joiner)
            start = truncate_len // 2
            end = start + len(joiner)
            string_shown = self.string[:start] + joiner + self.string[-end:]
        string_shown = repr(string_shown)  ## handle single and double quotes
        return f"{self.__class__.__name__}({string_shown})"
