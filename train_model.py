import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer


#USAGE: python train_model.py

#With more time, I would clean-up the code a bit more, spit it into methods and add more documentation
def main():
    tweet_path = '/Users/jmooney/Downloads/tweets.csv'
    random_tweet_path = '/Users/jmooney/Documents/tweets_random_all.csv'
    isis_tweets = pd.read_csv(tweet_path, names = ['name', 'username', 'description', 'location', 'followers',
           'numberstatuses', 'time', 'tweets'])
    random_tweets = pd.read_csv(random_tweet_path, names = ['tweet', 'content'])
    tweet_path = '/Users/jmooney/Downloads/tweets.csv'
    random_tweet_path = '/Users/jmooney/Documents/tweets_random_all.csv'
    isis_tweets = pd.read_csv(tweet_path)

    random_tweets = pd.read_csv(random_tweet_path)
    random_tweets['is_isis'] = 0
    isis_tweets['is_isis'] = 1
    isis_tweets= isis_tweets.rename(columns={'tweets':'content'})
    isis_tweets = isis_tweets[['content', 'is_isis']]
    random_tweets = random_tweets[['content', 'is_isis']]
    all_tweets = isis_tweets.append(random_tweets)

    BUFFER_SIZE = 1000
    BATCH_SIZE = 32

    X = all_tweets[['content']]
    y = all_tweets[['is_isis']]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.25)

    train_text = X_train['content'].to_numpy()
    train_labels = y_train.to_numpy().flatten()
    valid_text = X_test['content'].to_numpy()
    valid_labels = y_test.to_numpy().flatten()

    # CONVERT TO TF DATASETS

    train_ds = tf.data.Dataset.from_tensor_slices((train_text,train_labels))
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_text,valid_labels))

    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    valid_ds = valid_ds.batch(BATCH_SIZE)


    # PREFETCH

    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



    classifier_model = build_classifier_model()


    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    epochs = 3
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
    classifier_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)


    history = classifier_model.fit(x=train_ds,
                                   validation_data=valid_ds,
                                   epochs=epochs)

    loss, accuracy = classifier_model.evaluate(test_ds)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    dataset_name = 'threat_level'
    saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

    classifier_model.save(saved_model_path, include_optimizer=False)


def build_classifier_model():
    #Electra pretrained flavor of bert is pretrained as a descriminator. 
    #Starting with the small version but could potentially test larger 
    #networks in the future
    tfhub_handle_encoder = 'https://tfhub.dev/google/electra_small/2'
    tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

main()
