import argparse
from dataclasses import dataclass
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    use_chat_template: bool
    seed: int


@dataclass(frozen=True)
class AnalysisResult:
    # [num_layers+1, seq_len]
    argmax_tokens: np.ndarray
    # [seq_len]
    final_tokens: np.ndarray
    # [seq_len] first layer index where argmax matches final_tokens[t]
    emergence_layers: np.ndarray
    # Optional: rank curves
    rank_curves: Optional[np.ndarray]  # [num_rank_positions, num_layers+1]
    # Which token positions were ranked (these are input positions t)
    rank_positions: Optional[List[int]]


def load_model(model_name: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _maybe_apply_chat_template(
    tokenizer, prompt: str, use_chat_template: bool
) -> torch.Tensor:
    if not use_chat_template:
        enc = tokenizer(prompt, return_tensors="pt")
        return enc.input_ids

    # For chat-tuned models, prefer the built-in chat template if present.
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            # IMPORTANT: normalize the return type to a torch.Tensor.
            # Some tokenizers return tokenized objects instead of plain tensors.
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tokenizer(rendered, return_tensors="pt")
            return enc.input_ids
        except Exception:
            # If the chat template path fails for any reason, fall back to plain text.
            enc = tokenizer(prompt, return_tensors="pt")
            return enc.input_ids

    # Fallback: treat prompt as plain text.
    enc = tokenizer(prompt, return_tensors="pt")
    return enc.input_ids


def generate_sequence(
    model,
    tokenizer,
    prompt: str,
    cfg: GenerationConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, str]:
    torch.manual_seed(cfg.seed)
    input_ids = _maybe_apply_chat_template(
        tokenizer, prompt, cfg.use_chat_template
    ).to(device)

    tokenizer_pad = tokenizer.pad_token_id
    if tokenizer_pad is None:
        tokenizer_pad = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            do_sample=False,  # greedy
            max_new_tokens=cfg.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer_pad,
            eos_token_id=tokenizer.eos_token_id,
        )

    out_ids = out[0].detach().cpu()
    text = tokenizer.decode(out_ids, skip_special_tokens=False)
    return out_ids.unsqueeze(0), text


def run_forward(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
):
    input_ids = input_ids.to(device)
    with torch.no_grad():
        # Teacher forcing in one pass: feed the full sequence at once.
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    # hidden_states: tuple length num_layers+1, each [batch, seq_len, hidden_dim]
    hidden_stack = torch.stack(hidden_states, dim=0)
    # Shape: [num_layers+1, batch, seq_len, hidden_dim]
    return hidden_stack, outputs


def token_id_to_display(token_id: int, tokenizer, max_chars: int = 6) -> str:
    # convert_ids_to_tokens is stable and typically returns short sub-token strings.
    tok = tokenizer.convert_ids_to_tokens(int(token_id))
    # Llama often uses U+2581 (▁) to indicate whitespace. Make it readable in ASCII.
    tok = tok.replace("\u2581", "_").replace("\n", " ")
    if len(tok) > max_chars:
        tok = tok[:max_chars]
    return tok


def analyze_layers(
    model,
    tokenizer,
    hidden_states: torch.Tensor,
    rank_positions: Optional[List[int]] = None,
) -> AnalysisResult:
    """
    For each hidden state layer h_l[t], apply the final LM head and compute:
      - argmax token at each (t, l)
      - emergence layer: first l where argmax matches the final-layer argmax token
    Optionally compute rank curves for selected token positions t.
    """
    # hidden_states: [num_layers+1, batch, seq_len, hidden_dim]
    num_stages, batch, seq_len, _hidden_dim = hidden_states.shape
    assert batch == 1, "This script currently assumes batch size 1."

    lm_head = model.get_output_embeddings()
    # argmax over vocab for each (layer, position)
    argmax_tokens = np.zeros((num_stages, seq_len), dtype=np.int64)

    # emergence_layers[t] is the first layer where argmax==final_argmax[t]
    emergence_layers = np.full((seq_len,), fill_value=-1, dtype=np.int64)

    rank_curves = None
    if rank_positions:
        rank_curves = np.zeros((len(rank_positions), num_stages), dtype=np.int64)

    # 1) Compute layer-wise argmax via (softmax -> argmax) to match requirements.
    with torch.no_grad():
        for l in range(num_stages):
            h_l = hidden_states[l]  # [1, seq_len, hidden_dim]
            logits = lm_head(h_l)  # [1, seq_len, vocab]
            probs = torch.softmax(logits, dim=-1)  # [1, seq_len, vocab]
            top_ids = probs.argmax(dim=-1)  # [1, seq_len]
            argmax_tokens[l] = top_ids[0].detach().cpu().numpy()

    final_tokens = argmax_tokens[num_stages - 1].copy()  # [seq_len]

    # 2) Optional rank curves, computed after we know the final-layer argmax token ids.
    if rank_positions:
        with torch.no_grad():
            for l in range(num_stages):
                h_l = hidden_states[l][:, rank_positions, :]  # [1, K, hidden_dim]
                logits = lm_head(h_l)[0].detach().cpu().numpy()  # [K, vocab]
                for i, t in enumerate(rank_positions):
                    target_id = int(final_tokens[t])
                    target_logit = logits[i, target_id]
                    rank = int(np.sum(logits[i] > target_logit) + 1)
                    rank_curves[i, l] = rank

    # Emergence: first layer index where argmax equals final_tokens[t]
    for t in range(seq_len):
        target = final_tokens[t]
        for l in range(num_stages):
            if argmax_tokens[l, t] == target:
                emergence_layers[t] = l
                break

    return AnalysisResult(
        argmax_tokens=argmax_tokens,
        final_tokens=final_tokens,
        emergence_layers=emergence_layers,
        rank_curves=rank_curves,
        rank_positions=rank_positions,
    )


def visualize_token_grid(
    argmax_tokens: np.ndarray,
    final_tokens: np.ndarray,
    tokenizer,
    max_seq_len: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (14, 8),
    title: str = "",
):
    """
    Plot a grid [seq_len x num_layers] where each cell contains the argmax token.
    Green if the intermediate-layer token matches the final-layer token, else red.
    """
    # argmax_tokens: [num_stages, seq_len]
    num_stages, full_seq_len = argmax_tokens.shape

    if max_seq_len is not None and full_seq_len > max_seq_len:
        seq_len = max_seq_len
        argmax_tokens = argmax_tokens[:, :seq_len]
        final_tokens = final_tokens[:seq_len]
    else:
        seq_len = full_seq_len

    # match_grid: [seq_len, num_stages]
    match_grid = (argmax_tokens[:, :seq_len] == final_tokens[None, :seq_len]).T
    # Numeric grid for coloring: 1 match -> green, 0 mismatch -> red
    color_grid = match_grid.astype(np.int32)

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#d62728", "#2ca02c"])  # red, green

    # Font fallback to reduce missing glyph warnings for CJK tokens.
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    with warnings.catch_warnings():
        # Common warning when overlaying token text that your current font lacks.
        warnings.filterwarnings(
            "ignore", message=r"Glyph .* missing from font", category=UserWarning
        )
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(color_grid, aspect="auto", interpolation="nearest", cmap=cmap)

    # X: layers (0..num_stages-1), Y: token positions (0..seq_len-1)
    ax.set_xlabel("Layer (0 = embeddings, final = lm head source)")
    ax.set_ylabel("Token position (t)")
    if title:
        ax.set_title(title)

    # Annotate each cell with the token string
    for t in range(seq_len):
        for l in range(num_stages):
            tok_id = int(argmax_tokens[l, t])
            tok_str = token_id_to_display(tok_id, tokenizer)
            is_match = bool(match_grid[t, l])
            text_color = "black" if is_match else "white"
            ax.text(
                l,
                t,
                tok_str,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    # Make it easier to read without fully cluttering tick labels.
    ax.set_xticks(np.arange(0, num_stages, max(1, num_stages // 8)))
    ax.set_yticks(np.arange(0, seq_len, max(1, seq_len // 8)))
    ax.grid(False)

    # Add a simple legend by drawing two patches.
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="none", label="Matches final token"),
        Patch(facecolor="#d62728", edgecolor="none", label="Differs from final token"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_rank_curves(
    rank_curves: np.ndarray,
    rank_positions: List[int],
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Rank vs layer (for final token at each position)",
):
    # rank_curves: [num_rank_positions, num_stages]
    num_rank_positions, num_stages = rank_curves.shape
    x = np.arange(num_stages)

    plt.figure(figsize=(12, 6))
    for i in range(num_rank_positions):
        plt.plot(x, rank_curves[i], label=f"t={rank_positions[i]}")
    plt.yscale("log")
    plt.xlabel("Layer index (0 = embeddings, final = last hidden state)")
    plt.ylabel("Rank (lower is more probable)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def parse_rank_positions(s: Optional[str], seq_len: int) -> Optional[List[int]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    positions = []
    for p in parts:
        t = int(p)
        if t < 0 or t >= seq_len:
            raise ValueError(f"rank position t={t} out of range [0, {seq_len-1}]")
        positions.append(t)
    # Remove duplicates while preserving order
    dedup = []
    seen = set()
    for t in positions:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
    return dedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--dtype", default="auto", type=str, choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--plot_max_seq_len", default=48, type=int)
    parser.add_argument("--save_plot_dir", default="plots", type=str)
    parser.add_argument("--no_save_plots", action="store_true", help="Disable saving plots to disk")
    parser.add_argument("--no_show_plots", action="store_true", help="Disable showing plots via plt.show()")
    parser.add_argument(
        "--show_font_warnings",
        action="store_true",
        help="Show matplotlib font glyph warnings (usually noisy).",
    )
    parser.add_argument(
        "--rank_positions",
        default=None,
        type=str,
        help="Comma-separated input positions t for which to plot rank vs layer. Example: 0,1,5",
    )
    args = parser.parse_args()

    if not args.show_font_warnings:
        warnings.filterwarnings(
            "ignore", message=r"Glyph .* missing from font", category=UserWarning
        )

    # Device/dtype
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dtype == "auto":
        if device.type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        use_chat_template=(not args.no_chat_template),
        seed=args.seed,
    )

    model, tokenizer = load_model(args.model_name, device=device, dtype=dtype)
    generated_ids, generated_text = generate_sequence(
        model, tokenizer, args.prompt, cfg=gen_cfg, device=device
    )

    seq_len = generated_ids.shape[1]
    print("=== Generated text (greedy) ===")
    print(generated_text)
    print()

    # Forward pass once with teacher forcing
    hidden_states, outputs = run_forward(model, generated_ids, device=device)
    num_stages = hidden_states.shape[0]
    num_layers = num_stages - 1
    print(f"Hidden states shape: {tuple(hidden_states.shape)} (stages = embeddings + {num_layers} blocks)")

    rank_positions = parse_rank_positions(args.rank_positions, seq_len=seq_len) if args.rank_positions else None
    analysis = analyze_layers(
        model,
        tokenizer,
        hidden_states=hidden_states,
        rank_positions=rank_positions,
    )

    # Final-layer greedy argmax tokens
    final_token_strs = [token_id_to_display(t, tokenizer) for t in analysis.final_tokens]
    print("=== Final-layer greedy argmax tokens (by position t) ===")
    print(" ".join(final_token_strs))
    print()

    # Emergence per token position t:
    # Note: at input position t, the model predicts token at t+1 (causal LM shift).
    print("=== Emergence layers per position t (predicting t+1) ===")
    # Ignore the last position in this report since it predicts token at seq_len (out of range).
    for t in range(seq_len - 1):
        final_tok_id = int(analysis.final_tokens[t])
        final_tok = token_id_to_display(final_tok_id, tokenizer)
        emerge_l = int(analysis.emergence_layers[t])
        print(f"t={t:3d} -> final token '{final_tok}' emerges at layer {emerge_l}")
    print()

    # Quick consistency check: final argmax at t should match the actually generated token at t+1.
    # (This is the expected BOS/input shift mentioned in the prompt.)
    generated_next = generated_ids[0, 1:].detach().cpu().numpy()
    final_vs_next = analysis.final_tokens[: seq_len - 1]
    matches = int(np.sum(final_vs_next == generated_next))
    print(f"Final-layer argmax matches generated next tokens for {matches}/{seq_len-1} positions.")
    print()

    save_plots = not args.no_save_plots
    show_plots = not args.no_show_plots
    plot_dir = args.save_plot_dir
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    # Visualization grid (may truncate for readability)
    grid_save_path = (
        os.path.join(plot_dir, "token_emergence_grid.png") if save_plots else None
    )
    visualize_token_grid(
        analysis.argmax_tokens,
        analysis.final_tokens,
        tokenizer=tokenizer,
        max_seq_len=args.plot_max_seq_len,
        save_path=grid_save_path,
        show=show_plots,
        title=f"Layer-wise token emergence (TinyLlama) | showing first {min(seq_len, args.plot_max_seq_len)} positions",
    )

    # Optional: rank curves
    if analysis.rank_curves is not None and analysis.rank_positions is not None:
        rank_save_path = (
            os.path.join(plot_dir, "rank_curves.png") if save_plots else None
        )
        plot_rank_curves(
            analysis.rank_curves,
            analysis.rank_positions,
            save_path=rank_save_path,
            show=show_plots,
        )


if __name__ == "__main__":
    main()

