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
