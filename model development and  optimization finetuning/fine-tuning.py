from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
import evaluate, numpy as np, torch
import warnings

dataset = load_dataset('glue','sst2') #sentiment positive/negative dataset , dict like DatasetDict with data splits

model_name = "distilbert-base-uncased" # fast BERT variant , loads matching tokemizer , tok_fn maps raw "sentence" to tokenIds with truncation (keep sequence length reasonable for  GPU / CPU memory)
tok = AutoTokenizer.from_pretrained(model_name)
def tok_fn(b): return tok(b["sentence"],truncation=True)
data= dataset.map(tok_fn,batched=True,remove_columns=["sentence","idx"])

id2label = {0:"negative",1:"positive"}
label2id ={"negative":0,"positive":1}
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,num_labels=2,id2label=id2label,label2id=label2id
)
# freeze base initially(feature extraction)
# train only classification head
for p in model.base_model.parameters():
    p.requires_grad = False

acc = evaluate.load("accuracy")
def metrics(eval_pred):
    logits,labels=eval_pred
    preds = np.argmax(logits,axis=-1)
    return {"accuracy": acc.compute(predictions=preds,references=labels)["accuracy"]}

args = TrainingArguments(
    output_dir= "sst2-distilbert",
    learning_rate=5e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    eval_strategy="epoch",
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=42
)

train_ds = data["train"].shuffle(seed=42).select(range(8000))

trainer=Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=data["validation"],
    processing_class=tok,
    data_collator=DataCollatorWithPadding(tokenizer=tok),
    compute_metrics=metrics
)

print("WARM-up (head only):")
trainer.train()

# unfreeze top transformer layers
for p in model.base_model.parameters():
    p.requires_grad=False
for name,param  in model.base_model.named_parameters():
    if any(k in name for k in ["transformer.layer.4","transformer.layer.5"]):
        param.requires_grad=True  #unfreeze top layers

# lower LR for gengle finetune
fine_args= TrainingArguments(
    **{**args.to_dict(),"learning_rate":2e-5,"num_train_epochs":1}
)

#rebuild trainer
trainer = Trainer(
    model = model , args=fine_args,train_dataset=train_ds,
    eval_dataset=data["validation"], processing_class=tok,
    data_collator=DataCollatorWithPadding(tokenizer=tok),
    compute_metrics=metrics
)

print("FINE-TUNE (top layers + head):")
trainer.train()

print("Eval:",trainer.evaluate())

"""
Phase1:
Head only training freeze base ,train head high LR. => transfer the generic language representations learned by DistilBERT and train a small task specific classifier 

Phase2:
Top Layers + head unfreeze top 2 layers Low LR => unfreezing of the top layers after head alignment.

start with frozen transfer learning then move to partial finetuning
"""

"""
- you dont need to unfreeze all layers, you can get substantiative results by only training the few layers compared to full finetuning, if dataset is small,
training fewer parameters reduces the overfitting risk
- when unfreezing ,unfreeze gradually and monitor to see if it helps . this is called gradual unfreezing
- when layers are unfrozen , you dont have to use the same learning rate across the model , wiith discriminative finetuning,
earlier layers get a smaller learning rate while later layers classifier get a higher one, this protects lower level features like embeddings and allowing higher layers to adapt quickly

- always use smaller learning rate when finetuning the pretrained backbone, this avoids overwriting the useful knowledge encoded during pretraining 
- early stopping is often used in finetuning because its easy to overfit the small dataset. validation performance degrades you stop . 
- data augmentation and regularization can also help if finetuning data is limited
"""

