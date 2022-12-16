import tweepy
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import tweepy
import datetime
import time
import json

bootstrap_servers = "localhost:9092"

#### create topic and producer####
topic_name = "raw-tweets"
admin_client = KafkaAdminClient(
    bootstrap_servers=bootstrap_servers, 
    client_id='YC'
)
try:
    topic_list = []
    topic_list.append(NewTopic(name=topic_name, num_partitions=1, replication_factor=1))
    admin_client.create_topics(new_topics=topic_list, validate_only=False)
except:
    print(f"Topic {topic_name} already exists.")
admin_client.close()

producer = KafkaProducer(bootstrap_servers=bootstrap_servers, key_serializer=lambda k: json.dumps(k).encode('utf-8'),
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'), client_id="producer_raw_tweets")
                         
#### hear from streaming tweets and ####
token = r"AAAAAAAAAAAAAAAAAAAAAORWkAEAAAAAkKkTq5jo9zsToxDq5vs3a3qD%2F6o%3D73jcJ7TxJPOY8vg0gcGzB3QSWZyOIJGrF35jY6PEFCpkIvT0T6"
client = tweepy.Client(bearer_token=token)

start_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=60)
end_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=10)
query = 'covid'
tweet_fields = ['created_at', 'lang', 'possibly_sensitive']
max_results = 10

while True:
    # we search the tweets about "covid" in the past two minutes, maximum 100 pieces of tweets
    tweets = client.search_recent_tweets(query=query,
                                        tweet_fields= tweet_fields, 
                                        max_results=max_results, 
                                        start_time=start_time,
                                        end_time=end_time)
    start_time = end_time
    end_time = start_time + datetime.timedelta(seconds=10)

    #### a producer writing to kafka topics
    for i, tweet in enumerate(tweets.data):
        producer.send(topic_name, key=tweet.lang, value=tweet.text)
        print(f'Successfully send the {i}th tweet to Kafka! Language is ' + tweet.lang)
    print('waiting for 10 secondes!')
    time.sleep(10)
