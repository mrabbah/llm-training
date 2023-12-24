from transformers import AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor, AutoProcessor, AutoModelForSequenceClassification, AutoModelForTokenClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")