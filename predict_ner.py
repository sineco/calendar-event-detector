from transformers import AutoTokenizer, pipeline

# Path to your fine-tuned model
MODEL_PATH = "raraujo/bert-finetuned-ner"  

# Load the tokenizer and pipeline
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
ner_pipeline = pipeline("token-classification", model=MODEL_PATH, tokenizer=tokenizer, aggregation_strategy="simple")

# Example input text for predictions
sample_text = "Let's schedule a meeting on Thursday at 10 AM in the Zoom room."

# Get predictions
results = ner_pipeline(sample_text)
print(results)
# Display predictions
print(f"Input Text: {sample_text}")
print("Predictions:")
for entity in results:
    print(f"Entity: {entity['entity_group']}, Text: {entity['word']}, Start: {entity['start']}, End: {entity['end']}, Score: {entity['score']}")
