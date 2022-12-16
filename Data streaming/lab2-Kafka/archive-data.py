import json
from kafka import KafkaConsumer

topic_raw = "raw-tweets"
topic_en = "en-tweets"
topic_fr = "fr-tweets"
topic_pos = "positive-tweets"
topic_neg = "negative-tweets"
list_topics = [topic_raw, topic_en, topic_fr, topic_pos, topic_neg]
bootstrap_servers = "localhost:9092"
consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
consumer.subscribe(list_topics)

for message in consumer:
    text = json.loads(message.value)
    topic = message.topic
    if topic in list_topics:
        with open('./archive_files/'+topic+'.txt', 'a', encoding='utf-8') as f:
            f.write(text)
            f.write('\n')
        print(f"Archive a tweet of topic {topic} to {topic}.txt!")