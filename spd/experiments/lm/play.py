# %%
import torch
from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS
from transformers import AutoTokenizer

from spd.experiments.lm.models import EmbeddingComponent, LinearComponentWithBias, SSModel

# %%
# Select the model size you want to use
model_size = "1.25M"  # Options: "35M", "30M", "11M", "5M", "1.25M"

# Load model configuration
model_config = MODEL_CONFIGS[model_size]

# Load appropriate model
model_path = f"chandan-sreedhara/SimpleStories-{model_size}"
model = Llama.from_pretrained(model_path, model_config)
# model.to("cuda")
model.eval()

# %%
# Load a pythia model instead
from transformers import AutoModelForCausalLM

pythia_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
pythia_model.eval()
# %%
from transformers import AutoModelForCausalLM

# Now load the tinystories model
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.eval()

# %%
# Get the maximum cosine similarity for each row of the embedding matrix
emb = model.transformer.wte.weight.clone().to("cuda")
# emb = pythia_model.gpt_neox.embed_in.weight.clone().to("cuda")
normed_emb = emb / emb.norm(dim=-1, keepdim=True)
x = normed_emb @ normed_emb.T
# Remove the diagonal
x[torch.arange(x.shape[0]), torch.arange(x.shape[0])] = 0
max_cosine_sims = x.max(dim=-1).values
print("Shape of max cosine sims", max_cosine_sims.shape)
print(max_cosine_sims[:50])
print("Max cosine sim", max_cosine_sims.max())
print("Mean cosine sim", max_cosine_sims.mean())
# %%

ss_model = SSModel(
    llama_model=model,
    target_module_patterns=["model.transformer.h.*.mlp.gate_proj"],
    m=17,
    n_gate_hidden_neurons=None,
)

# # Create components with rank=10 (adjust as needed)
# gate_proj_components = create_target_components(
#     model, rank=m, target_module_patterns=["model.transformer.h.*.mlp.gate_proj"]
# )
gate_proj_components: dict[str, LinearComponentWithBias | EmbeddingComponent] = {
    k.removeprefix("components.").replace("-", "."): v for k, v in ss_model.components.items()
}  # type: ignore
# %%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

# Define your prompt
prompt = "The curious cat looked at the"

# IMPORTANT: Use tokenizer without special tokens
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
# input_ids = inputs.input_ids.to("cuda")
input_ids = inputs.input_ids
# Targets should be the inputs shifted by one (we will later ignore the last input token)
targets = input_ids[:, 1:]
input_ids = input_ids[:, :-1]

# IMPORTANT: Set correct EOS token ID (not the default from tokenizer)
eos_token_id = 1

# %%

# # Generate text
# with torch.no_grad():
#     output_ids = model.generate(
#         idx=input_ids, max_new_tokens=20, temperature=0.7, top_k=40, eos_token_id=eos_token_id
#     )

# # Decode output
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(f"Generated text:\n{output_text}")


# %%

# logits, _ = ss_model.forward(input_ids, components=gate_proj_components)
logits, _ = ss_model.forward(input_ids)
print("inputs_shape", input_ids.shape)
print("logits", logits)
print("logits shape", logits.shape)

logits, _ = ss_model.forward_with_components(input_ids, components=gate_proj_components)

print("Component logits shape", logits.shape)
print("Component logits", logits)

# Create some dummy masks
masks = {
    f"model.transformer.h.{i}.mlp.gate_proj": torch.randn(1, input_ids.shape[-1], ss_model.m)
    for i in range(len(model.transformer.h))
}

logits, _ = ss_model.forward_with_components(
    input_ids, components=gate_proj_components, masks=masks
)

print("Masked component logits shape", logits.shape)
print("Masked component logits", logits)
#########################################################
# %%
