"""
Logit Lens using TransformerBridge (TransformerLens 3.0 API)
=============================================================
TransformerLens 3.0 introduced TransformerBridge as the new recommended way
to load models.  HookedTransformer is now DEPRECATED.

  OLD: model = HookedTransformer.from_pretrained("gpt2")
  NEW: model = TransformerBridge.boot_transformers("gpt2")

Both support the same run_with_cache() / hook API.
"""

import torch
from transformer_lens.model_bridge import TransformerBridge

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load model
print("Loading GPT-2 via TransformerBridge ...")
model = TransformerBridge.boot_transformers("SecCoderX/Qwen2.5_Coder_7B_SecCoderX_aligned")
model.eval()

print(f"Model: {model.cfg.model_name}  |  Layers: {model.cfg.n_layers}  |  d_model: {model.cfg.d_model}")
print()

paris_id = model.tokenizer.encode(" Paris")[0]
print(f"Token ID for ' Paris': {paris_id}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# WHY GPT-2 doesn't say "Paris" for "The capital of France is":
#
# GPT-2 is a TEXT COMPLETION model, NOT a Q&A model.
# It predicts the statistically most likely next word from internet text.
#
# On the internet, "The capital of France is" is almost always followed by
# descriptive words: "now", "the", "currently", "home to" — NOT just "Paris".
#
# Example sentences that dominate training data:
#   "The capital of France is NOW one of the most visited cities..."
#   "The capital of France is THE hub of European fashion..."
#
# To get Paris, we need a prompt where a city name is the most
# statistically natural completion. Let's compare three prompts:
# ─────────────────────────────────────────────────────────────────────────────

prompts = [
    # Prompt 1: statement form — what we tried before
    ("Statement (original)", "The capital of France is"),

    # Prompt 2: Q&A format forces a noun/city as the answer
    ("Q&A format", "Q: What is the capital of France?\nA:"),

    # Prompt 3: "Paris is the capital of France." reversed
    ("Fill-in-the-blank", "France's capital city is"),
]


def logit_lens_for_prompt(prompt_name, prompt):
    print(f"\n{'='*65}")
    print(f"  PROMPT [{prompt_name}]: \"{prompt}\"")
    print(f"{'='*65}")

    logits, cache = model.run_with_cache(prompt)

    # --- Full model final prediction ---
    final_probs = torch.softmax(logits[0, -1, :], dim=-1)
    top5 = torch.topk(final_probs, 5)

    print("\n  Final output")
    for i, (tid, p) in enumerate(zip(top5.indices, top5.values), 1):
        tok = model.tokenizer.decode(tid.item())
        marker = " <-- PARIS!" if tid.item() == paris_id else ""
        print(f"    #{i}: {tok!r:<20} {p:.2%}{marker}")

    sorted_ids = torch.argsort(final_probs, descending=True)
    paris_rank_final = (sorted_ids == paris_id).nonzero(as_tuple=True)[0].item() + 1
    paris_prob_final = final_probs[paris_id].item()
    print(f"\n  ' Paris' final rank: #{paris_rank_final}  prob: {paris_prob_final:.2%}")

    # --- Logit Lens: layer by layer ---
    print(f"\n  {'Layer':>5}  {'Top-1':>20}  {'Paris rank':>12}  {'Paris prob':>10}")
    print(f"  {'-'*55}")

    for layer in range(model.cfg.n_layers):
        # Residual stream after this transformer block
        h_l = cache[f"blocks.{layer}.hook_resid_post"]

        # Apply the final LayerNorm (CRITICAL: W_U expects normalized input)
        h_l_normed = model.ln_final(h_l)

        # Project to vocabulary via unembedding matrix W_U
        logits_l = model.unembed(h_l_normed)

        # Probabilities at the last token position
        probs = torch.softmax(logits_l[0, -1, :], dim=-1)

        # Top-1 token
        top1_id = torch.argmax(probs).item()
        top1_tok = model.tokenizer.decode(top1_id)
        top1_p = probs[top1_id].item()

        # Paris rank
        sorted_l = torch.argsort(probs, descending=True)
        paris_rank = (sorted_l == paris_id).nonzero(as_tuple=True)[0].item() + 1
        paris_p = probs[paris_id].item()

        star = " ***" if paris_rank <= 5 else ""
        label = f"{top1_tok!r}({top1_p:.1%})"
        print(f"  {layer:>5}  {label:>20}  {'#'+str(paris_rank):>12}  {paris_p:>10.2%}{star}")


for name, prompt in prompts:
    logit_lens_for_prompt(name, prompt)

