import json
from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import re
from textblob import TextBlob

# Text cleaning function
def cleanTweet(tweet: str) -> str:
    tweet = re.sub(r'http\S+', '', str(tweet))
    tweet = re.sub(r'bit.ly/\S+', '', str(tweet))
    tweet = tweet.strip('[link]')
    # Remove users
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))
    # Remove punctuation
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@â'
    tweet = re.sub('[' + my_punctuation + ']+', ' ', str(tweet))
    # Remove numbers
    tweet = re.sub('([0-9]+)', '', str(tweet))
    # Remove hashtags
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))
    # Remove extra simbols
    tweet = re.sub('@\w+', '', str(tweet))
    tweet = re.sub('\n', '', str(tweet))
    return tweet

# Assign sentiment to tweets
def Sentiment(tweet: str) -> str:
    if TextBlob(tweet).sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Positive'


bootstrap_servers = "localhost:9092"
topic_en = "en-tweets"
topic_fr = "fr-tweets"
topic_pos = "positive-tweets"
topic_neg = "negative-tweets"
admin_client = KafkaAdminClient(
    bootstrap_servers=bootstrap_servers, 
    client_id='YC'
)
for topic in [topic_pos, topic_neg]:
    try:
        topic_list = [NewTopic(name=topic, num_partitions=1, replication_factor=1)]
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
    except:
        print(f"Topic {topic} already exists.")
admin_client.close()

consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
consumer.subscribe([topic_en, topic_fr])

producer = KafkaProducer(bootstrap_servers=bootstrap_servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                        client_id="producer_sentiments")

for message in consumer:
    text = json.loads(message.value)
    clean_text = cleanTweet(text)
    if Sentiment(clean_text) == 'Positive':
        producer.send(topic_pos, clean_text)
        print(f'Successfully send a tweet to topic "positive_tweets"!')
    elif Sentiment(clean_text) == 'Negative':
        producer.send(topic_neg, clean_text)
        print(f'Successfully send a tweet to topic "negative_tweets"!')
    else:
        raise ValueError("Fail to write to positive tweets nor to negative tweets!")
    