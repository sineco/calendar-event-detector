import os
import tensorflow as tf
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TFAutoModelForTokenClassification
from transformers.keras_callbacks import PushToHubCallback
import evaluate

print(tf.__version__)

# Using this because I don't have GPU
#export TF_ENABLE_MIXED_PRECISION=0

os.environ["TOKENIZERS_PARALLELISM"] = "false"



data_files = {"train": "./data/generated_train.json",
              "test": "./data/generated_test.json", 
              "validation": "./data/generated_eval.json"}

raw_datasets =  load_dataset("json", data_files=data_files )

# Define the label set
LABEL_SET = ['O', 'B-PLATFORM', 'I-PLATFORM', 'B-TIME', 'I-TIME', 'B-DATE', 'I-DATE']

# Rename some columns
raw_datasets = raw_datasets.rename_column("ner_tags", "labels")
raw_datasets = raw_datasets.rename_column("tokens", "words")

# Tokenize the model


model_checkpoint = "bert-base-cased" #"bert-base-cased" 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print(tokenizer.is_fast)

inputs = tokenizer(raw_datasets["train"][0]["words"], is_split_into_words=True)
print(inputs.tokens())
print(inputs.word_ids())

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# Test alignment
# Example debugging step
sample_idx = 0
sample_words = raw_datasets["train"][sample_idx]["words"]
sample_labels = raw_datasets["train"][sample_idx]["labels"]
aligned_inputs = tokenizer(sample_words, is_split_into_words=True)
aligned_labels = align_labels_with_tokens(sample_labels, aligned_inputs.word_ids())

print("Words:", sample_words)
print("Tokens:", aligned_inputs.tokens())
print("Word IDs:", aligned_inputs.word_ids())
print("Labels:", sample_labels)
print("Aligned Labels:", aligned_labels)


tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, return_tensors="tf"
)

batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
print(batch["labels"]) 

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16,
)

tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)

id2label = {i: label for i, label in enumerate(LABEL_SET)}
label2id = {v: k for k, v in id2label.items()}


model = TFAutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

print(model.config.num_labels)

from transformers import create_optimizer
import tensorflow as tf

# Train in mixed-precision float16
# Comment this line out if you're not? using a GPU that will not benefit from this
#tf.keras.mixed_precision.set_global_policy("mixed_float16")

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_epochs = 6
num_train_steps = len(tf_train_dataset) * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

callback = PushToHubCallback(output_dir="bert-finetuned-ner", tokenizer=tokenizer)


model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=[callback],
    epochs=num_epochs,
)



metric = evaluate.load("seqeval")

labels = raw_datasets["train"][0]["labels"]
labels = [LABEL_SET[i] for i in labels]

# Create fake predictions
predictions = labels.copy()
predictions[2] = "O"
metric.compute(predictions=[predictions], references=[labels])

# Calculating the metrics of the real predictions
all_predictions = []
all_labels = []
for batch in tf_eval_dataset:
    logits = model.predict_on_batch(batch)["logits"]
    labels = batch["labels"]
    predictions = np.argmax(logits, axis=-1)
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(LABEL_SET[predicted_idx])
            all_labels.append(LABEL_SET[label_idx])
metric.compute(predictions=[all_predictions], references=[all_labels])
print(metric.compute(predictions=[all_predictions], references=[all_labels]))

from sklearn.metrics import classification_report

# Generate a classification report
print(classification_report(all_labels, all_predictions))

# Replace this with your own checkpoint
from transformers import pipeline
model_checkpoint = "bert-finetuned-ner"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
print("Res:" , token_classifier("what time are we joining the google meeting call?"))

sentence = "What time is the Google meeting?"
results = token_classifier(sentence)
print("Results:", results)


##########################
# Tokenize test dataset
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels, batched=True, remove_columns=raw_datasets["test"].column_names
)

# Prepare collator and evaluation dataset
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)

# Load evaluation metric
seqeval = evaluate.load("seqeval")

# Collect predictions and true labels
all_predictions = []
all_labels = []

for batch in tf_test_dataset:
    logits = model.predict_on_batch(batch)["logits"]
    labels = batch["labels"]
    predictions = np.argmax(logits, axis=-1)

    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:  # Skip special tokens
                continue
            all_predictions.append(LABEL_SET[predicted_idx])
            all_labels.append(LABEL_SET[label_idx])

# Compute metrics
results = seqeval.compute(predictions=[all_predictions], references=[all_labels])
print("SeqEval Metrics:", results)

# Generate a classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, labels=LABEL_SET))


############
# Get the first test sample
single_sample = raw_datasets["test"][1]

# Tokenize the sample
tokenized_sample = tokenizer(
    single_sample["words"],
    truncation=True,
    is_split_into_words=True,
    return_tensors="tf"
)

# Predict using the model
logits = model(tokenized_sample)["logits"]
predictions = tf.argmax(logits, axis=-1)

# Map predictions back to labels
predicted_labels = [LABEL_SET[pred] for pred in predictions.numpy()[0]]

# Display the results
print("Tokens:", single_sample["words"])
print("True Labels:", [LABEL_SET[label] for label in single_sample["labels"]])
print("Predicted Labels:", predicted_labels)
