import gc
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from transformers import DataCollatorForSeq2Seq
from transformers.adapters import IA3Config
from transformers import TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback
from transformers import Seq2SeqTrainingArguments
from datasets import load_dataset
from transformers import Seq2SeqTrainer

def add_template(examples):
    for i,ex in enumerate(examples['Context']):
        examples['Context'][i] = instruction + "\nContext: " + ex + "\n" + examples["Question"] + "\nAnswer:"
    return examples

def preprocess_function(examples, model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    inputs = [ex for ex in examples["Context"]]
    targets = [ex for ex in examples["Answer"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length= 512, truncation=True
    )
    return model_inputs



def model_trainer(file_path, model_name, adapter_name, instruction, save_path):


    df = pd.read_csv(file_path)

    df.to_csv("train_data.csv", sep = ",")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    pytorch_total_params = sum(torch.numel() for torch in model.parameters())
    print(f"Total Parameters: {pytorch_total_params}")

    config = IA3Config()
    model.add_adapter(adapter_name, config=config)

    model.train_adapter(adapter_name)
    model.train()

    pytorch_total_params = sum(torch.numel() for torch in model.parameters() if torch.requires_grad)
    print(f"Total Parameters: {pytorch_total_params}")

    dataset = load_dataset('csv', data_files = {
            "train": "train_data.csv"
            })

    dataset = dataset.remove_columns("Unnamed: 0")

    instruction = instruction

    dataset = dataset.map(add_template, batched = True)
    
    inputs = [ex for ex in dataset["Context"]]
    targets = [ex for ex in dataset["Answer"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length= 512, truncation=True
    )

    tokenized_datasets = model_inputs

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


    args = Seq2SeqTrainingArguments(
    save_path,
    #evaluation_strategy="epoch",
    #save_strategy = "epoch",
    learning_rate = 3e-3,
    logging_steps = 1,
    per_device_train_batch_size = 5,
    per_device_eval_batch_size = 5,
    weight_decay=0.01,
    num_train_epochs= 15,
    predict_with_generate=True,
    #load_best_model_at_end = True,
    #metric_for_best_model = "eval_loss",
    greater_is_better = False,
    logging_strategy = "epoch",
    no_cuda = False,
    save_total_limit = 2,
    #fp16=True, #Uncomment to use GPU
    #fp16_full_eval = True
    )

    trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets,
    #eval_dataset=tokenized_datasets["validation"], #Uncomment if you have validation dataset
    data_collator=data_collator,
    tokenizer=tokenizer,
    #compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(save_path)

    return f"The model is saved at {save_path}"

dataset = "section_5.csv"
model_name = "google/flan-t5-xl"
adapter_name = "section_5_adapter"
ins = "Instruction: Read the question in the Context below and answer the Question."
save_path = "section_5/flan-t5-xl"


result = model_trainer(dataset, model_name, adapter_name, ins, save_path)
print(result)
