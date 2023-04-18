# backprompt

`backprompt` provides a data structure which allows a user to dynamically construct
prompts while avoiding repeated LLM computations.

<img src="meme.jpg" alt="meme" width="400"/>

<br>

⚠️ **This is an in-progress, toy project.** It currently only works with locally loaded
GPT-2 models from HuggingFace. It doesn't try to be all that useful.

It would be cool if this idea could be used to create a prompt representation database.
A key is the (text, its children, an ID for the weights ID, an ID for the tokenizer) and
the value is the model's representation of the text conditional on its children.


## Motivation

In many large-scale tasks performed by LLMs, a particular prompt is used many times—once
for each instance of the task. In cases like these, the amount of computation performed
by future LLM calls can be reduced by caching and re-using the LLM's representation of
the prompt.

`backprompt` takes this well-known idea a step further by additionally caching LLM
representations of *intermediate* text in the prompt. Intermediate caching may be useful
when one needs to dynamically adjust the prompt without having to re-compute the LLM's
representation of it. `backprompt` effectively abstracts the complex process of prompt
construction and caching as plain-old string concatenation, which everyone and their
grandpa knows how to do :-)


## Usage

TODO: perhaps the API endpoint classification problem 


## Installation

```
python -m pip install git+https://github.com/kddubey/backprompt
```


## How it works

If you basically know how [backprop](https://en.wikipedia.org/wiki/Backpropagation)
works (watch [this YouTube video](https://www.youtube.com/watch?v=VMj-3S1tku0)), and you
basically know how a decoder-only autoregressive language model works (watch [this
YouTube video](https://www.youtube.com/watch?v=kCc8FmEb1nY)), then you know how
`backprompt` works :-)

Analogies:
  - (backprop &rarr; "intermediate" gradient of a function)<br>
    (`backprompt` &rarr; attention block keys and values).
  - (backprop &rarr; gradient of a function)<br>
    (`backprompt` &rarr; token logits).
  - (backprop &rarr; chain rule)<br>
    (`backprompt` &rarr; tensor concatenation).

TODO: graph visualization


## Testing

TODO: expand test cases

```
pytest
```


## Todos

<details>
<summary>Research</summary>

- [ ] Demonstrate time-savings, compute memory costs
- [ ] What's the computational complexity of using past keys and values wrt # tokens?
- [ ] What's it gonna take to create a DB?

</details>

<details>
<summary>Code</summary>

- [ ] Test
- [ ] Graph visualization
- [ ] Allow for frozen representations / custom independencies in the graph
- [ ] Flesh out README
- [ ] Batching
- [ ] Eager mode
- [ ] `ModelRepr` dataclass for convenience
    - [ ] Add and update a `token_logprobs` attribute to the LM output obj
    - [ ] By default, only keep last (non-pad) token's logits in the LM output obj
- [ ] Documentation?

</details>
