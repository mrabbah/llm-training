from transformers import pipeline
import torch

generator = pipeline('text-generation', model='distilgpt2', device=0)
result = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=2)
print(result)

# pipe = pipeline("text-classification", device=0)
# result = pipe(["This restaurant is awesome", "This restaurant is awful"])
# print(result)

# transcriber = pipeline(task="automatic-speech-recognition", device=0)
# result = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
# print(result)

# transcriber = pipeline(model="openai/whisper-large-v2", device_map="auto")
# result = transcriber([
#         "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
#         "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
#     ])
# print(result)

#This runs the pipeline on the 4 provided audio files, but it will pass them in batches of 2 to the model (which is on a GPU, where batching is more likely to help) 
# All tasks provide task specific parameters which allow for additional flexibility and options to help you get your job done. 
# transcriber = pipeline(model="openai/whisper-large-v2", device_map="auto", batch_size=2, return_timestamps=True)
# audio_filenames = [f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)]
# texts = transcriber(audio_filenames)
# print(texts)


# KeyDataset is a util that will just output the item we're interested in.
# from transformers.pipelines.pt_utils import KeyDataset
# from datasets import load_dataset

# pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
# dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

# for out in pipe(KeyDataset(dataset, "audio")):
#     print(out)


# def data():
#     for i in range(10):
#         yield f"My example {i}"


# pipe = pipeline(model="gpt2", device=0)
# generated_characters = ""
# for out in pipe(data()):
#     generated_characters += out[0]["generated_text"] + " \n--- "
# print(generated_characters)

# vision_classifier = pipeline(model="google/vit-base-patch16-224")
# # images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
# preds = vision_classifier(
#     images="https://images2.alphacoders.com/944/944438.jpg"
# )
# preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
# print(preds)





# This model is a `zero-shot-classification` model.
# It will classify text, except you are free to choose any label you might imagine
# classifier = pipeline(model="facebook/bart-large-mnli")
# result = classifier(
#     "I have a problem with my iphone that needs to be resolved asap!!",
#     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
# )
# print(result)

# vqa = pipeline(model="impira/layoutlm-document-qa")
# result = vqa(
#     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
#     question="What is the invoice number?",
# )
# print(result)

# pipe = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device_map="auto")
# output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
# print(output)
# # You can also pass 8-bit loaded models if you install bitsandbytes and add the argument load_in_8bit=True
# pipe = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"load_in_8bit": True})
# output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
# print(output)