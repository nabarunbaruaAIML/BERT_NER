[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FnabarunbaruaAIML%2FBERT_NER&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Introduction

This Repository was built to show case DistilBERT Implementation for NER which is built on Pytorch. We used distilbert-base-uncased Pretrained Model from Huggingface Transformer Library and Fine Tune it for token classification i.e. NER.

Since Guthub blocks, push for more than 100 MB's file (https://help.github.com/en/github/managing-large-files/conditions-for-large-files) therefore please download the Weights file from Google Drive ( https://drive.google.com/file/d/1MTdgl1qfOEo-TuT39ByrbQsFZQ-y7v3a/view?usp=sharing )

### Output
![image](https://user-images.githubusercontent.com/64695833/128056565-8886016b-4d95-4338-8be1-b52b99e0edb4.png)




## Run Locally

Clone the project

```bash
  git clone https://github.com/nabarunbaruaAIML/BERT_NER.git
```

Go to the project directory

```bash
  cd BERT_NER
```

Install dependencies

```bash
  conda env create -f environment.yml
```
All dependencies are included in the environment.yml file.

Start the server

```bash
  Python3 clientApp.py
```

  
