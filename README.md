# Text Summarization Model using Double Embedding

The project is focusing on text summarization of the high court judgment of New Zealand.

## Dependencies

```
Python>=3.7.0
torch>=0.5.0
```


## Embedding

```
1. Download Glove embedding (glove.6B.100d.txt): https://nlp.stanford.edu/projects/glove/

2. Download Law2Vec embedding (Law2Vec.100d.txt): https://archive.org/details/Law2Vec

```

## Training

```
python main.py --is_processed_data 1 --is_train 1
```

## Testing

```
python main.py --is_processed_data 0 --is_train 0
```
