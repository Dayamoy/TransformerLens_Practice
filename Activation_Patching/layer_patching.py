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

# === Step 1 & 2: Get clean and corrupted caches ===
clean_prompt = "The capital of France is"
corrupt_prompt = "The capital of Poland is"

clean_logits, clean_cache = model.run_with_cache(clean_prompt)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_prompt)

# Get token IDs for "Paris" and "Warsaw"
paris_id = model.to_single_token(" Paris")
warsaw_id = model.to_single_token(" Warsaw")

# Baseline logit differences (last token position)
clean_logit_diff = (clean_logits[0, -1, paris_id] - clean_logits[0, -1, warsaw_id]).item()
corrupt_logit_diff = (corrupt_logits[0, -1, paris_id] - corrupt_logits[0, -1, warsaw_id]).item()
print(f"Clean logit diff:   {clean_logit_diff:.2f}")
print(f"Corrupt logit diff: {corrupt_logit_diff:.2f}")

# === Step 3 & 4: Patch each layer and measure effect ===
results = []
for layer in range(model.cfg.n_layers):
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Create a hook function that replaces corrupted with clean
    # IMPORTANT: Only patch the LAST token position.
    # The model predicts the next token from position -1, so patching all
    # positions would trivially restore the full clean context at every layer,
    # giving effect=1.0 everywhere (which is the bug you observed).
    def patch_hook(activation, hook, layer=layer):
        # activation shape: [batch, seq_len, d_model]
        # Only restore the residual stream at the final token position
        print(f"patch_hook Layer {layer}: ")
        activation[:, -1, :] = clean_cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
        return activation

    # Run corrupted prompt WITH the patch applied
    patched_logits = model.run_with_hooks(
        corrupt_prompt,
        fwd_hooks=[(hook_name, patch_hook)]
    )

    # Measure: did the output shift toward "Paris"?
    patched_logit_diff = (
        patched_logits[0, -1, paris_id] - patched_logits[0, -1, warsaw_id]
    ).item()

    # Normalize: 0 = no effect, 1 = fully restored clean answer
    effect = (patched_logit_diff - corrupt_logit_diff) / (clean_logit_diff - corrupt_logit_diff)
    results.append(effect)
    print(f"Layer {layer:2d}: effect = {effect:.3f}")