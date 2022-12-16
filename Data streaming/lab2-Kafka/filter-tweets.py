import json
from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic

bootstrap_servers = "localhost:9092"

topic_listening = "raw-tweets"
#### create topics ####
topic_en = "en-tweets"
topic_fr = "fr-tweets"
admin_client = KafkaAdminClient(
    bootstrap_servers=bootstrap_servers, 
    client_id='YC'
)
for topic in [topic_en, topic_fr]:
    try:
        topic_list = [NewTopic(name=topic, num_partitions=1, replication_factor=1)]
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
    except:
        print(f"Topic {topic} already exists.")
admin_client.close()


consumer = KafkaConsumer(
    topic_listening,
    bootstrap_servers=bootstrap_servers,
    client_id="consumer_from_raw"
)
producer = KafkaProducer(bootstrap_servers=bootstrap_servers, client_id="producer_tweets_2langs")
while True:
    for message in consumer:
        lang = json.loads(message.key)
        if lang == 'en':
            producer.send(topic_en, message.value)
            print(f'Successfully send a tweet to topic "en_tweets"!')
        elif lang == 'fr':
            producer.send(topic_fr, message.value)
            print(f'Successfully send a tweet to topic "fr_tweets"!')
        else:
            print(f'Tweet language {lang}, do not send message.')

