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
    topic = message.topic
    partition = message.partition
    offset = message.offset
    timestamp = message.timestamp
    print(f"Topic: {topic} ---- Partition: {partition} ---- Offset: {offset} ---- Timestamp: {timestamp}.")