from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor
from datasets import load_dataset, Audio

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
# print(encoded_input)
# decoded_tokens = tokenizer.decode(encoded_input["input_ids"])
# print(decoded_tokens)

# batch_sentences = [
#     "But what about second breakfast?",
#     "Don't think he knows about second breakfast, Pip.",
#     "What about elevensies?",
# ]
# encoded_inputs = tokenizer(batch_sentences)
# print(encoded_inputs)

# batch_sentences = [
#     "But what about second breakfast?",
#     "Don't think he knows about second breakfast, Pip.",
#     "What about elevensies?",
# ]
# encoded_input = tokenizer(batch_sentences, padding=True)
# print(encoded_input)

# batch_sentences = [
#     "Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about Don't think he knows about second breakfast, Pip. what about second breakfast? and what abo ut second breakfast? can you tell me what about second breakfast? yes i can tell you what about second breakfast. what about select breakfast? do you know what about second breakfast? in my opinion, we should talk about second breakfast. what about giving new git a try? what about",
# ]
# encoded_input = tokenizer(batch_sentences, padding=True, truncation=False)
# print(len(encoded_input["input_ids"][0]))
# encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
# print(len(encoded_input["input_ids"][0]))

# batch_sentences = [
#     "But what about second breakfast?",
#     "Don't think he knows about second breakfast, Pip.",
#     "What about elevensies?",
# ]
# encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
# print(encoded_input)

# def preprocess_function(examples):
#     audio_arrays = [x["array"] for x in examples["audio"]]
#     inputs = feature_extractor(
#         audio_arrays,
#         sampling_rate=16000,
#         padding=True,
#         max_length=100000,
#         truncation=True,
#     )
#     return inputs

# dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
# print(dataset[0]["audio"])
# # For this tutorial, you’ll use the Wav2Vec2 model. Take a look at the model card, and you’ll learn Wav2Vec2 is pretrained on 16kHz sampled speech audio
# # Use 🤗 Datasets’ cast_column method to upsample the sampling rate to 16kHz:
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
# print(dataset[0]["audio"])
# # Next, load a feature extractor to normalize and pad the input. When padding textual data, a 0 is added for shorter sequences. 
# # The same idea applies to audio data. The feature extractor adds a 0 - interpreted as silence - to array.
# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
# # Pass the audio array to the feature extractor. We also recommend adding the sampling_rate argument in the feature extractor in order to better debug any silent errors that may occur.
# audio_input = [dataset[0]["audio"]["array"]]
# features = feature_extractor(audio_input, sampling_rate=16000)
# print(features)
# print(dataset[0]["audio"]["array"].shape)
# print(dataset[1]["audio"]["array"].shape)
# processed_dataset = preprocess_function(dataset[:5])
# print(processed_dataset["input_values"][0].shape)
# print(processed_dataset["input_values"][1].shape)

# Multimodal
lj_speech = load_dataset("lj_speech", split="train")
# For ASR, you’re mainly focused on audio and text so you can remove the other columns:
lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
print(lj_speech[0]["audio"])
print(lj_speech[0]["text"])
# Remember you should always resample your audio dataset’s sampling rate to match the sampling rate of the dataset 
# used to pretrain a model!
lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
def prepare_dataset(example):
    audio = example["audio"]
    example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))
    return example
prepared_ds = prepare_dataset(lj_speech[0])
print(prepared_ds)
