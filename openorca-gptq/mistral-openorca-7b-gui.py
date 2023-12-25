import os
from threading import Thread
from typing import Iterator

import gradio as gr

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DESCRIPTION = "🐋 Mistral-7B-OpenOrca 🐋"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"


MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

MODEL_ID = "TheBloke/Mistral-7B-OpenOrca-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
device = "cuda" # if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=device) 

def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    # conversation = [{"role": "system", "content": "I'm a Corteza Blocks Coder! I will assist you in crafting powerful and efficient components for the Corteza low code platform. I'm expert in JavaScript, TypeScript, Vue.js, Bootstrap, and Bootstrap-Vue, I can generate code snippets, suggest best practices, or answer any questions you may have about Corteza low code development!"}]
    conversation = []
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})
    print(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
    generation_config = model.generation_config
    generation_config.max_new_tokens=max_new_tokens
    generation_config.do_sample=True
    generation_config.top_p=top_p
    generation_config.top_k=top_k
    generation_config.temperature=temperature
    generation_config.repetition_penalty=repetition_penalty
    generation_config.use_exllama=False
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        generation_config=generation_config,
        num_beams=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.2,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=None,
    examples=[
        ["Hello there! How are you doing?"],
        ["Can you explain briefly to me what is the Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["How many hours does it take a man to eat a Helicopter?"],
        ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ],
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()

if __name__ == "__main__":
    demo.queue(max_size=20).launch()