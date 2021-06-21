from flask import Flask
from flask import jsonify
from flask import request
import json
import config
import tweepy
import train_model
import tensorflow as tf

app = Flask(__name__)
server_location = config.SERVER_LOCATION
saved_model_path = server_location + '/twitter_terror_threat_indicator/threat_level_bert'
model = tf.keras.models.load_model(saved_model_path)
# model = tf.saved_model.load_model(saved_model_path)

@app.route("/")
def index():
    print("PRINTING")
    return json.dumps({"This is not a valid endpoint": "try adding /threat_level/[tweet_id]"})

# Endpoint for tweet analysis. receives tweet id from request path directly and uses it to 
# obtain the tweet content, feed it to the threat assesment model, and return a threat score
# from 0 to 1. 0 being unlikely pro-ISIS author and 1 being very likely a pro-ISIS author
@app.route('/threat_level/<tweet_id>', methods = ['GET'])
def get_threat_level(tweet_id):
    if request.method == 'GET':
        print("this was a GET request")
        score = calculate_threat(tweet_id)
        return jsonify({"That tweet's author is this likely to be pro-ISIS": score})


def calculate_threat(tweet_id):
    examples = []
    tweet = get_tweet(tweet_id)
    examples.append(tweet)
    results = tf.sigmoid(model(tf.constant(examples)))
    return results


def get_tweet(tweet_id):
    # 
    API_KEY = config.API_KEY
    API_SECRET_KEY = config.API_SECRET_KEY
    BEARER_TOKEN = config.BEARER_TOKEN
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
    api = tweepy.API(auth)

    tweet = api.get_status(tweet_id)
    return tweet.text

if __name__ == '__main__':
    app.run()