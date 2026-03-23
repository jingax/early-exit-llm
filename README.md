# Layer-wise Token Emergence (TinyLlama)

This repo contains a single script, `layerwise_token_emergence.py`, which analyzes *when* each token becomes the model’s greedy prediction across layers in a decoder-only LLM.

It does the following:
1. Loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` via `AutoModelForCausalLM` and `AutoTokenizer`
2. Generates a full continuation using `model.generate` with greedy decoding (`do_sample=False`)
3. Runs one teacher-forced forward pass with `output_hidden_states=True`
4. For every layer (embedding through final hidden state), applies the model’s final LM head to each token position and computes the argmax token
5. Reports the **first layer** where the intermediate-layer argmax matches the **final-layer** argmax
6. Visualizes a token grid with green/red cells and overlays the token text

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python layerwise_token_emergence.py --prompt "Hello, explain early exit in one sentence." --max_new_tokens 32
```

Optional (chat template control, rank curves, and plot truncation):

```bash
python layerwise_token_emergence.py --prompt "Tell me a joke." --max_new_tokens 32 --no_chat_template
python layerwise_token_emergence.py --prompt "Explain attention." --max_new_tokens 24 --rank_positions 0,1,5
python layerwise_token_emergence.py --prompt "Describe transformers." --max_new_tokens 32 --plot_max_seq_len 40
```

## Interpreting `t` (causal shift)

For a causal LM, the logits at input position `t` predict the **next token** at position `t+1`.

This script prints emergence layers per `t` (i.e., “the layer where the prediction for token `t+1` stabilizes”).

## Notes

The visualization overlays token text into every cell, so it can be slow and cluttered for long sequences. Use `--plot_max_seq_len` to cap how many token positions are shown in the plot (the analysis still runs on the full generated sequence).

