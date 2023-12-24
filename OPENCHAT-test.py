# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TheBloke/openchat_3.5-GPTQ")
model = AutoModelForCausalLM.from_pretrained("TheBloke/openchat_3.5-GPTQ", device_map="cuda")