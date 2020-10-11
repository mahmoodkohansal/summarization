# Automatic Text Summarization System

This repository contains some source codes that I developed in my Master Degree thesis. My thesis was about single text summarization in Persian but all of my approaches doesn't depends on any language and can apply to all languages. I used Word2Vec word embedding to score sentences and clustering method to choose sentences from scrored sentences. 

It has several python files that developed for testing several approaches through completing the thesis. Therefore many of this python files are not usable or not applied in the best approach I found in my final thesis.


## Usage

For using the final approach that gains me the best results, you can use ```summarizer_service.py``` file. It has a complete flow of summarization process and can be run with below command. 

```
python3 summarizer_service.py
```

A Flask API will be run by this command that can be used with a POST request to summarize a new text.

```
[POST] http://localhost:5000/summarize
Post Body:
  - input : The text you want to summarize
  - len : lengthe of summarization text you want in character
```

## TODO
Readme is not completed and many useful codes for Summarization or other NLP tasks are developed here that needs more discription.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)