from transformers import AutoModelForCausalLM, AutoTokenizer #, AutoConfig
import torch

MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# config = AutoConfig.from_pretrained(MODEL_ID)
# print(config)

device = "cuda" # if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=device) 

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)

generation_config = model.generation_config
print(generation_config)
generation_config.max_new_tokens=1000
generation_config.do_sample=True

generated_ids = model.generate(model_inputs, generation_config=generation_config)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
