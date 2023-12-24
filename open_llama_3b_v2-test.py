# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
# model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2", device_map="cuda")

tokenizer = AutoTokenizer.from_pretrained("TheBloke/open-llama-3b-v2-wizard-evol-instuct-v2-196k-GPTQ")
model = AutoModelForCausalLM.from_pretrained("TheBloke/open-llama-3b-v2-wizard-evol-instuct-v2-196k-GPTQ", device_map="cuda")
