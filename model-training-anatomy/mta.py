import numpy as np
from datasets import Dataset
from pynvml import *
from transformers import TrainingArguments, Trainer, logging, AutoModelForSequenceClassification, AutoTokenizer
import torch

seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

print_gpu_utilization()

torch.ones((1, 1)).to("cuda")
print_gpu_utilization()

MODEL_ID = "TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ"
# MODEL_ID = "Felladrin/TinyMistral-248M-SFT-v4"
# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_ID, 
#     torch_dtype=torch.float16,
#     use_flash_attention_2=True).to("cuda")
device = "cuda" # if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, device_map=device) 
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to("cuda")
print_gpu_utilization()

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = model.config.eos_token_id

logging.set_verbosity_error()
max_seq_length = 2048

training_args = TrainingArguments(per_device_train_batch_size=4, optim="adafactor", gradient_checkpointing=True, fp16=True, **default_args)
trainer = Trainer(model=model, args=training_args, train_dataset=ds, tokenizer=tokenizer)
result = trainer.train()
print_summary(result)

