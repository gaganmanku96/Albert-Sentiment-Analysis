# ALBERT for Sentiment Analysis

## Dataset preparation
A tab seperated (.tsv) file is required with the name of train i.e. train.tsv
Train dataset needs to be placed in a folder.

## How to fine-tune
#### Following parameters are required
1. --data_dir - Directory where data is stored
2. --model_type - The model which we wanna use for fine-tuning. Here, we are using <i>albert</i>
3. --model_name_or_path - The variant of albert which you want to use.
4. --output_dir - path where you want to save the model.
5. --do_train - because we are training the model.

#### Example
```
$ python run_glue.py --data_dir data --model_type albert --model_name_or_path albert-base-v2 --output_dir output --do_train
```

## Different Models available for use
|                | Average  | SQuAD1.1 | SQuAD2.0 | MNLI     | SST-2    | RACE     |
|----------------|----------|----------|----------|----------|----------|----------|
|V2              |
|ALBERT-base     |82.3      |90.2/83.2 |82.1/79.3 |84.6      |92.9      |66.8      |
|ALBERT-large    |85.7      |91.8/85.2 |84.9/81.8 |86.5      |94.9      |75.2      |
|ALBERT-xlarge   |87.9      |92.9/86.4 |87.9/84.1 |87.9      |95.4      |80.7      |
|ALBERT-xxlarge  |90.9      |94.6/89.1 |89.8/86.9 |90.6      |96.8      |86.8      |
|V1              |
|ALBERT-base     |80.1      |89.3/82.3 | 80.0/77.1|81.6      |90.3      | 64.0     |
|ALBERT-large    |82.4      |90.6/83.9 | 82.3/79.4|83.5      |91.7      | 68.5     |
|ALBERT-xlarge   |85.5      |92.5/86.1 | 86.1/83.1|86.4      |92.4      | 74.8     |
|ALBERT-xxlarge  |91.0      |94.8/89.3 | 90.2/87.4|90.8      |96.9      | 86.5     |
(table taken from Google-research)

## Prediction
Both docker and python file are available for prediction.
1. Set the name of folder where model files are stored.
2. Run api.py file
```
$ python api.py
```
or
```
from api import SentimentAnalyzer
classifier = SentimentAnalyzer()
print(classifier.predict('the movie was nice'))
```

## Thanks to HuggingFace for making the implementation simple and also Google for this awesome pretrained model.
