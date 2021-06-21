# Twitter Terror Threat Indicator
------------------
This project implements a basic flask server and python ML model to return a score from 0 to 1 regarding how likely a user is to be pro-isis.  

## Model Used
It uses BERT, a Pre-trained of Deep Bidirectional Transformer for Language Understanding. More specifically, it uses a BERT-like model with pretrained weights that is specifically trained to be a text discriminator. This BERT-like model is called Electra.  

ELECTRA is trained as a discriminator in a set-up resembling a generative adversarial network (GAN). It was originally published in 2020.
https://openreview.net/attachment?id=r1xMH1BtvB&name=original_pdf  

In order to make the checkpoint compatible, the dense layer (pooler) on top of the CLS token (which is non-existent in the original pretraining) was initialized with the identity matrix.  

Since this is a binary classification problem and the model outputs a probability (a single-unit layer), I used a Binary Crossentropy loss function.  

Since it was both important to catch pro-ISIS tweets, and not falsely accuse someone of being pro-ISIS. If I had more time, I would have also taken a look at the ROC AUC score and potentially I could have altered the fine-tuning to focus more on detecting True Positives than True Negatives.  

With more time, I would also test the results of Electra-large. In the published results of the Electra model, Electra-large significantly outperforms both smaller versions of itself, vanilla BERT, GPT2/3 and more.  

I would also implement a productionized serving system.  

I ended up having to train on less than all the data. The training time was over 20 hours for the full data set so I ended up sampling from the training data and using this smaller data set. With more time, I would use all data available.



## Error Rate Acheived



## Requirements
python 3.6
A stable version of tensorflow
  

## Set-up 

Apply for twitter API credentials here: https://developer.twitter.com/en/apply-for-access
If you already have twitter API credentials you may skip with step.

After cloning, and obtaining credentials, create a config.py file in the route directory. 
populate the file with the following, replacing the values with your credentials.

EXAMPLE (DO NOT USE):  
API_KEY = 'abcdefghijklmnop'  
API_SECRET_KEY = '123456789abcd987654321'  
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAJsnvsjvnsjkdbnvdfjkslvncasfjkvnsalnvajefksvnjslkfmbnjadkfsvksjdfnjsf'  
SERVER_LOCATION = '/Users/jmooney/Documents/personal_admin/interviews/activefence'

`cd twitter_terror_threat_indicator`

`source venv/bin/activate`

`pip install -r requirments.txt`


## Usage
### Train
`source venv/bin/activate`  
  
`python train_model.py`  

  
### Predict
`source venv/bin/activate`

`flask run`

After running the flask server from your terminal, check what port it is being run on and navigate to it on a web browser of your choice.

You can hit the endpoint by simply going to wherever the server is running + '/threat_level/<tweet_id>' for example http://127.0.0.1:5000/threat_level/440322224407314432.

This will trigger a GET request and return a json-like response including the text from the tweet.

The saved fine-tuned version of BERT will then be loaded and used to predict how likely the tweet's author is to be pro-ISIS.


## Resources
Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2018.

Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning: ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators, ICLR 2020.

Paper titles: *An Effective BERT-Based Pipeline for Twitter Sentiment
Analysis: A Case Study in Italian*

Classify text with BERT
Tutorial: https://www.tensorflow.org/text/tutorials/classify_text_with_bert

Pretrained BERT-like discriminator model: https://tfhub.dev/google/electra_small/2

Text preprocessing for BERT: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3


## Model Summary:
Total params: 13,549,058
Trainable params: 13,549,057
Non-trainable params: 1

loss: 0.0525 - binary_accuracy: 0.9839