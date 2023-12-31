from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablecode-completion-alpha-3b")
model = AutoModelForCausalLM.from_pretrained(
  "stabilityai/stablecode-completion-alpha-3b",
  trust_remote_code=True,
  torch_dtype="auto",
)
model.cuda()
inputs = tokenizer("import torch\nimport torch.nn as nn", return_tensors="pt").to("cuda")
tokens = model.generate(
  input_ids=inputs,
  max_new_tokens=4096,
  temperature=0.2,
  do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
