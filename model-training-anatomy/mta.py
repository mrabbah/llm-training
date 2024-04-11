import numpy as np
from datasets import load_dataset
from pynvml import *
from transformers import TrainingArguments, Trainer, logging, AutoModelForSequenceClassification, AutoTokenizer, TrainerCallback
import torch
from sklearn.preprocessing import LabelEncoder
import os
import time
os.environ['TORCH_CHECKPOINT_USE_REENTRANT'] = 'False'  # This option saves memory because it assumes that the graph will not be re-entered during the backward pass. It’s suitable for simpler computational graphs where reentrancy is not needed. This setting can reduce memory usage but might lead to errors or incorrect gradients for complex models requiring reentrant backward passes.

# utilities function for monitoring progress:
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    total_steps = len(train_dataset)
    current_step = trainer.state.global_step
    progress_percentage = (current_step / total_steps) * 100
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print(f"Training progress: {progress_percentage:.2f}%")
    print_gpu_utilization()

class TimeCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.last_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        if current_time - self.last_time >= 60:  # 60 seconds (1 minute)
            self.last_time = current_time
            trainer, _ = kwargs['model'], kwargs['logs']
            print_summary(trainer, state) 

print_gpu_utilization()

# loading dataset
ds = load_dataset("papluca/language-identification")
ds.set_format("pt")
print(ds)
print(ds["train"][0])


# labels are in string format example "en", "fr", "de", "ar"... we must convert them to integers
# before feeding them to the model
label_encoder = LabelEncoder()
label_encoder.fit(ds["train"]["labels"])
print(label_encoder.classes_)
print(label_encoder.transform(label_encoder.classes_))

# loading the model & the tokenizer
MODEL_ID = "Felladrin/TinyMistral-248M-SFT-v4"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, 
    device_map=device, 
    num_labels=len(label_encoder.classes_)
) 

print_gpu_utilization()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = model.config.eos_token_id

logging.set_verbosity_error()

# Tokenizing the dataset
def tokenize(batch):
    tokenized_inputs = tokenizer(batch["text"], padding=True, truncation=True)
    tokenized_inputs["labels"] = label_encoder.transform(batch["labels"])
    return tokenized_inputs

train_dataset=ds["train"].map(tokenize, batched=True)
eval_dataset=ds["validation"].map(tokenize, batched=True)
test_dataset=ds["test"].map(tokenize, batched=True)

# Setting up the training arguments
default_args = {
    "output_dir": "./tmp",
    "logging_dir": './logs',
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

training_args = TrainingArguments(
    per_device_train_batch_size=4, 
    optim="adafactor", 
    gradient_checkpointing=True, 
    fp16=True, 
    **default_args
)

# Training the model
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[TimeCallback()]
)
result = trainer.train()
print_summary(result)

# Perform evaluation on the test dataset
eval_result = trainer.evaluate()
print(f"Evaluation result: {eval_result}")



