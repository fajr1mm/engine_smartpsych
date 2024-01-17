import pandas as pd
from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import concatenate_datasets


app = FastAPI()

@app.route('/train')
def training(dataset):
    model_id="google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    nltk.download("punkt")

    metric = evaluate.load("f1")
    token = tokenize(datatraim)
    tokenized_dataset = datatraim.map(preprocess_function, batched=True, remove_columns=['RESPONSE', 'LEVEL KOMPETENSI'])
    
    label_pad_token_id = -100
    
    postprocess_text(preds, labels)
    compute_metrics(eval_preds)
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=3e-4,

        num_train_epochs=2,
        # logging & evaluation strategies
        logging_dir=f"{local_output_dir}/logs",
        logging_strategy="epoch",
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="tensorboard",
        push_to_hub=False,  # Tidak push ke Hugging Face Hub
    )
    
    trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    local_model_dir = "./model"
    model.save_pretrained(local_model_dir)
    tokenizer.save_pretrained(local_model_dir)


def tokenize_dataset(dataset):
    model_id="google/flan-t5-base"
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["RESPON"], truncation=True), batched=True, remove_columns=['RESPON', 'LEVEL KOMPETENSI'])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["LEVEL KOMPETENSI"], truncation=True), batched=True, remove_columns=['RESPON', 'LEVEL KOMPETENSI'])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = [item for item in sample["RESPON"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["LEVEL KOMPETENSI"], max_length=max_target_length, padding=padding, truncation=True)
    
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

    
