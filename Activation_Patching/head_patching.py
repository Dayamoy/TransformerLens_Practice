import torch
import numpy as np

# Patch each attention head individually
head_effects = np.zeros((model.cfg.n_layers, model.cfg.n_heads))

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        hook_name = f"blocks.{layer}.attn.hook_z"

        def patch_head_hook(activation, hook, layer=layer, head=head):
            # Only patch THIS specific head (dimension index 2)
            activation[:, :, head, :] = clean_cache[
                f"blocks.{layer}.attn.hook_z"
            ][:, :, head, :]
            return activation

        patched_logits = model.run_with_hooks(
            corrupt_prompt,
            fwd_hooks=[(hook_name, patch_head_hook)]
        )

        patched_diff = (
            patched_logits[0, -1, paris_id] -
            patched_logits[0, -1, warsaw_id]
        ).item()

        head_effects[layer, head] = (
            (patched_diff - corrupt_logit_diff) /
            (clean_logit_diff - corrupt_logit_diff)
        )

# Visualize as a heatmap
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(head_effects, cmap="RdBu", vmin=-0.3, vmax=0.3)
ax.set_xlabel("Head"); ax.set_ylabel("Layer")
ax.set_title("Activation Patching: Effect of Each Attention Head")
plt.colorbar(im, label="Normalized Patching Effect")
plt.tight_layout(); plt.savefig("head_patching.png", dpi=150)
plt.show()
