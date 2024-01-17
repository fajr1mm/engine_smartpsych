from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from preprocessing_function import clean_text
from flask import jsonify
from datetime import datetime
import os
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np  
import nltk
from nltk.tokenize import sent_tokenize


## TRAIN
class TrainItem(BaseModel):
    columns: List[str]
    parameters: Optional[dict] = {}
    data: List[Dict[str, str]]

def train_model(data: TrainItem, columns: List[str] = [], parameters: dict = {}):
    data_records = data.data

    df_train, df_test = train_test_split(pd.DataFrame.from_records(data_records), test_size=0.3, random_state=42)
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)

    # Create DatasetDict with the desired format
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })


    # Menghapus kolom '__index_level_0__' dari "train" dataset
    dataset["train"] = dataset["train"].remove_columns('__index_level_0__')

    # Menghapus kolom '__index_level_0__' dari "test" dataset
    dataset["test"] = dataset["test"].remove_columns('__index_level_0__')

    # Setel path untuk menyimpan model dan tokenizer
    save_dir = f"./model/v{datetime.now().strftime('%d%m%Y')}"
    os.makedirs(save_dir, exist_ok=True)

    # Load tokenizer dan model
    model_id = "google/flan-t5-base"    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


    filtered_columns = [col for col in columns if col != 'level']
    features_column = ','.join(filtered_columns)  # Replace with the actual features column name

    # Combine train and test datasets
    combined_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

    # Tokenize the specified features column
    tokenized_inputs = combined_dataset.map(
        lambda x: tokenizer(x[features_column], truncation=True),
        batched=True,
        remove_columns=[features_column]
    )

    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

    # Tokenize the specified level column
    tokenized_targets = combined_dataset.map(
        lambda x: tokenizer(x['level'], truncation=True),
        batched=True,
        remove_columns=[features_column]
    )

    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

    # print(columns)
    # Apply preprocess_function to the dataset
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(max_source_length, max_target_length, x, columns, tokenizer, padding="max_length"),
        batched=True,
        remove_columns=[features_column, 'level']
    )

    #passing value from paramters dictionary
    if parameters:
        num_train_epochs = parameters["num_train_epochs"]
        per_device_train_batch_size = parameters["per_device_train_batch_size"]
        per_device_eval_batch_size = parameters["per_device_eval_batch_size"]
        save_total_limit = parameters["save_total_limit"]
    else:
        num_train_epochs = 2
        per_device_train_batch_size = 8
        per_device_eval_batch_size = 8
        save_total_limit = 2

    # Argument pelatihan
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=True,
        learning_rate=3e-4,
        num_train_epochs=num_train_epochs,
        logging_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        push_to_hub=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )


    # Pelatihan model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )


    # Jalankan pelatihan
    print("START TRAIN")
    trainer.train()

    # Simpan model dan tokenizer menggunakan pickle
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return {"status": "Model trained and saved successfully", "model_dir": save_dir}



def preprocess_function(max_source_length, max_target_length, sample, columns, tokenizer, padding="max_length"):
    # Add prefix to the input for T5
    prefix = "please classify this statement:"
    # Concatenate selected columns into a single string
    inputs = [f"{prefix} {sample[col]}" for col in columns]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(sample["level"], max_length=max_target_length, padding=padding, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"][0] if len(labels["input_ids"]) > 0 else []
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels



def compute_metrics(eval_preds):
    print("compute start")
    model_id = "google/flan-t5-base"    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    print("compute end")