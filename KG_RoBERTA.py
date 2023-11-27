# %%
#!git clone https://github.com/RUCAIBox/HaluEval.git

# %%
#!git submodule update --recursive

# %%
# !pip install transformers[torch]

# %%


# %%
import pickle
# import graphvite as gv
#import dataset
import nltk
import re
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# %%
# !pip install transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# %%
from transformers import TrainingArguments, Trainer
from huggingface_hub import login

login(token='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')

# %%
#!pip install datasets

# %%
# !wget -O ~/scratch/processed_data.pkl https://www.dropbox.com/scl/fi/vp5c9hcyqnaottfre5yas/processed_data.pkl?rlkey=i0gko0chfwyj5fn4rhsx9o3ve&dl=0

# %%
import os

TRANSFORMERS_CACHE = os.path.expanduser("~/scratch/transformers_cache")
os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)

HF_DATASETS_CACHE = os.path.expanduser("~/scratch/hf_datasets_cache")
os.makedirs(HF_DATASETS_CACHE, exist_ok=True)

# %%
processed_file = open(os.path.expanduser("~/scratch/processed_data.pkl"), "rb")
graph_data = pickle.load(processed_file)

# %%
import json

# Opening JSON file
data_list = []
with open('submodules/HaluEval/data/qa_data.json') as json_file:
    for jsonObj in json_file:
        data_dict = json.loads(jsonObj)
        data_list.append(data_dict)



# %%
connection_dict = {obj["question"].strip().lower() : i for i, obj in enumerate(data_list)}

# %%
len(connection_dict)

# %%
from datasets import load_dataset

def get_data(data_path):

  ds = load_dataset("json", data_files=data_path, split="train", cache_dir=HF_DATASETS_CACHE)
  ds = ds.train_test_split(test_size=0.2)
  return ds

#print(os.listdir())

ds = get_data("submodules/HaluEval/data/qa_data.json")
print(ds)

# %%
import torch

def preprocess_function(examples):
  questions = [q.strip().lower() for q in examples["question"]]
  inputs = tokenizer(
      questions,
      [c.lower() for c in examples["knowledge"]],
      max_length=384,
      truncation="only_second",
      return_offsets_mapping=True,
      padding="max_length",
  )
  offset_mapping = inputs.pop("offset_mapping")
  answers = examples["right_answer"]
  start_positions = []
  end_positions = []
  graph_embeddings = []

  for i, offset in enumerate(offset_mapping):
      answer = answers[i]
      start_char = examples["knowledge"][i].find(answer)
      end_char = start_char + len(answer)
      sequence_ids = inputs.sequence_ids(i)

      # Find the start and end of the context
      idx = 0
      while sequence_ids[idx] != 1:
          idx += 1
      context_start = idx
      while sequence_ids[idx] == 1:
          idx += 1
      context_end = idx - 1

      # If the answer is not fully inside the context, label it (0, 0)
      if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
          start_positions.append(0)
          end_positions.append(0)
      else:
          # Otherwise it's the start and end token positions
          idx = context_start
          while idx <= context_end and offset[idx][0] <= start_char:
              idx += 1
          start_positions.append(idx - 1)

          idx = context_end
          while idx >= context_start and offset[idx][1] >= end_char:
              idx -= 1
          end_positions.append(idx + 1)

      corr_index = connection_dict[questions[i]]
      graph_index = graph_data[corr_index]
      embeddings = torch.zeros(384, 512)
      token_list = inputs.input_ids[i]
      for ent, embedding in graph_index["knowledge"].items():
          ent_token_list = [tokenizer(ent).input_ids[1:-1], tokenizer(" " + ent).input_ids[1:-1]]

          for i in range(len(token_list) - len(list(ent_token_list)[0])):
            if token_list[i:i+len(list(ent_token_list)[0])] in ent_token_list:
              embeddings[i:i+len(list(ent_token_list)[0]), :] = torch.broadcast_to(torch.from_numpy(embedding), (len(list(ent_token_list)[0]), 512))
      for ent, embedding in graph_index["question"].items():
          ent_token_list = [tokenizer(ent).input_ids[1:-1], tokenizer(" " + ent).input_ids[1:-1]]

          for i in range(len(token_list) - len(list(ent_token_list)[0])):
            if token_list[i:i+len(list(ent_token_list)[0])] in ent_token_list:
              embeddings[i:i+len(list(ent_token_list)[0]), :] = torch.broadcast_to(torch.from_numpy(embedding), (len(list(ent_token_list)[0]), 512))
      graph_embeddings.append(embeddings)

  inputs["start_positions"] = start_positions
  inputs["end_positions"] = end_positions
  inputs["graph_embeddings"] = graph_embeddings

  return inputs

# %%
ds["train"][1]

# %%
print(ds["train"][1]["question"].strip())
corr_index = connection_dict[ds["train"][1]["question"].strip().lower()]
print(corr_index)
#print(graph_data[corr_index])
graph_index = graph_data[corr_index]
#for ent, embedding in graph_index["knowledge"].items():
#    graph_embeddings.append(embedding)
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=TRANSFORMERS_CACHE)

for k, v in graph_index["question"].items():
  print("{}, {}, {}".format(k, tokenizer(k).input_ids[1:-1], tokenizer(" " + k).input_ids[1:-1]))
for k, v in graph_index["knowledge"].items():
  print("{}, {}, {}".format(k, tokenizer(k).input_ids[1:-1], tokenizer(" " + k).input_ids[1:-1]))
print(tokenizer("Teen Wolf"))
print(str(tokenizer.decode([664, 2143, 417])) + "," + str(tokenizer.decode([39556])))

# %%
example = {
    "knowledge": [ds["train"][1]["knowledge"]],
    "question": [ds["train"][1]["question"]],
    "right_answer": [ds["train"][1]["right_answer"]],
    "hallucinated_answer": [ds["train"][1]["hallucinated_answer"]]
}
#print(clean_text(example))
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=TRANSFORMERS_CACHE)
inputs = preprocess_function(example)
print(inputs)
print(tokenizer.decode(inputs["input_ids"][0]))

# %%
torch.device('cuda')

# %%
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=TRANSFORMERS_CACHE)
tokenized_ds = ds.map(preprocess_function, batched=True, remove_columns=ds["train"].column_names)

# %%
#print(tokenized_ds["train"][0])
print(tokenized_ds["train"][0]["graph_embeddings"])
print(len(tokenized_ds["train"][0]["graph_embeddings"]))
print(len(tokenized_ds["train"][0]["graph_embeddings"][0]))
# print(tokenized_ds["train"]["end_positions"][0:10])

# %%
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

# %%
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import QuestionAnsweringModelOutput

class GraphEnrichedRoberta(nn.Module):
  def __init__(self, model_name):
    super().__init__()
    self.llm_embedding_size = 768
    self.graph_embedding_size = 512
    self.llm = AutoModelForQuestionAnswering.from_pretrained(model_name)
    self.fc = nn.Linear(self.graph_embedding_size + self.llm_embedding_size, self.llm_embedding_size)

    for param in self.llm.roberta.encoder.parameters():
      param.requires_grad = False

  def forward(self, input_ids, attention_mask, start_positions, end_positions, graph_embeddings):
     outputs = self.llm.roberta(input_ids, attention_mask)
     sequence_outputs = outputs.last_hidden_state

     enriched_encodings = self.fc(torch.cat((sequence_outputs, graph_embeddings), 2))
     enriched_encodings = F.relu(enriched_encodings)

     logits = self.llm.qa_outputs(enriched_encodings)
     start_logits, end_logits = logits.split(1, dim=-1)
     start_logits = start_logits.squeeze(-1).contiguous()
     end_logits = end_logits.squeeze(-1).contiguous()

     total_loss = None
     if start_positions is not None and end_positions is not None:
          # If we are on multi-GPU, split add a dimension
         if len(start_positions.size()) > 1:
             start_positions = start_positions.squeeze(-1)
         if len(end_positions.size()) > 1:
             end_positions = end_positions.squeeze(-1)
          # sometimes the start/end positions are outside our model inputs, we ignore these terms
         ignored_index = start_logits.size(1)
         start_positions = start_positions.clamp(0, ignored_index)
         end_positions = end_positions.clamp(0, ignored_index)

         loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
         start_loss = loss_fct(start_logits, start_positions)
         end_loss = loss_fct(end_logits, end_positions)
         total_loss = (start_loss + end_loss) / 2

     return QuestionAnsweringModelOutput(
         loss=total_loss,
         start_logits=start_logits,
         end_logits=end_logits,
         hidden_states=outputs.hidden_states,
         attentions=outputs.attentions,
     )

# %%
model = GraphEnrichedRoberta(model_name)

# %%
with torch.no_grad():
  input_ids_test = torch.tensor(tokenized_ds["train"][0]["input_ids"]).unsqueeze(0)
  attn_mask_test = torch.tensor(tokenized_ds["train"][0]["attention_mask"]).unsqueeze(0)
  start_positions_test = torch.tensor(tokenized_ds["train"][0]["start_positions"]).unsqueeze(0)
  end_positions_test = torch.tensor(tokenized_ds["train"][0]["end_positions"]).unsqueeze(0)
  graph_embeddings_test = torch.tensor(tokenized_ds["train"][0]["graph_embeddings"]).unsqueeze(0)
  print(model(input_ids_test, attn_mask_test, start_positions_test, end_positions_test, graph_embeddings_test))

# %%
training_args = TrainingArguments(
    output_dir="roberta_qa_model_with_graph",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# %%
trainer.train()

# %%
torch.save(model.state_dict(), 'graph_model_weights.pth')

# %%
model.load_state_dict(torch.load('graph_model_weights.pth'))

# %%
model

# %%
#tokenized_ds["test"]

predictions = trainer.predict(tokenized_ds["test"].remove_columns(["graph_embeddings"]))

# %%
import numpy as np

for i in range(len(predictions.predictions)):
  print(predictions.predictions[i].shape)

print(predictions.predictions[0])

print(np.array(tokenized_ds["test"]["start_positions"]).shape)


# %%
avg_loss = np.mean(predictions.predictions[0])
avg_loss

# %%
start_logits = predictions.predictions[1]
end_logits = predictions.predictions[2]

start_positions = np.array(tokenized_ds["test"]["start_positions"])
end_positions = np.array(tokenized_ds["test"]["end_positions"])

# %%
start_positions_pred = np.argmax(start_logits, axis=1)
end_positions_pred = np.argmax(end_logits, axis=1)

# %%
print(start_positions_pred)
print(start_positions)
print(end_positions_pred)
print(end_positions)
print(np.sum(start_positions_pred == start_positions))
overlap = np.maximum(0.0, np.minimum(end_positions_pred, end_positions) - np.maximum(start_positions_pred, start_positions) + 1)
print(overlap)
val1 = np.abs(1.0 * overlap / (end_positions - start_positions + 1))
val2 = np.abs(1.0 * overlap / (end_positions_pred - start_positions_pred + 1))
print(np.min(val1))
print(np.max(val1))
print(np.min(val2))
print(np.max(val2))

# %%
def exact_match(start_logits, end_logits, start_positions, end_positions):
  start_positions_pred = np.argmax(start_logits, axis=1)
  end_positions_pred = np.argmax(end_logits, axis=1)
  same_start = start_positions_pred == start_positions
  same_end = end_positions_pred == end_positions
  return np.sum(np.logical_and(same_start, same_end)) / start_positions.shape[0]

# %%
exact_match(start_logits, end_logits, start_positions, end_positions)

# %%
def f1_score(start_logits, end_logits, start_positions, end_positions):
  start_positions_pred = np.argmax(start_logits, axis=1)
  end_positions_pred = np.argmax(end_logits, axis=1)
  overlap = np.maximum(0, np.minimum(end_positions_pred, end_positions) - np.maximum(start_positions_pred, start_positions) + 1)
  recall = np.abs(1.0 * overlap / (end_positions - start_positions + 1))
  precision = np.abs(1.0 * overlap / (end_positions_pred - start_positions_pred + 1))
  f1 = (2 * precision * recall) / (precision + recall + 0.00000001)
  return np.mean(f1)

# %%
f1_score(start_logits, end_logits, start_positions, end_positions)

# %%
import torch
from torch import nn
import torch.nn.functional as F

class GraphEnrichedRoberta(nn.Module):
  def __init__(self, model_name):
    super().__init__()
    self.llm_embedding_size = 768
    self.llm = AutoModelForQuestionAnswering.from_pretrained(model_name)

    for param in self.llm.roberta.encoder.parameters():
      param.requires_grad = False

  def forward(self, input_ids, attention_mask, start_positions, end_positions):
    return self.llm(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

# %%
model = GraphEnrichedRoberta(model_name)

# %%
model.load_state_dict(torch.load('model_weights.pth'))

# %%



