In this TP, I have experimented `Twitter Streaming API` using `python` and developed a streaming application that use `Kafka` as the caching framework. 

On Windows, I installed Kafka in Docker. Here is a simple [guide (in chinese)](https://zhuanlan.zhihu.com/p/591911792). In the repository, you can also find a `docker-compose.yml` file to install the Kafka and Zookeeper image in Docker.

As for the analyse of streaming tweets with `kafka-python` package, here are two tutorials: [one in French](https://larevueia.fr/classification-de-tweets-en-direct-avec-apache-kafka-et-tweepy/) and [one in English](https://medium.com/mcd-unison/twitter-sentiment-analysis-using-zookeeper-kafka-and-pyspark-live-streaming-on-windows-10-in-2022-ada7757097a2). Just for your information, you can consult the [API for kafka-python package](https://kafka-python.readthedocs.io/en/master/apidoc/modules.html).

# What do these scripts do?

- The script `ingest-tweets.py` gets streaming tweets based on the keyword "covid" (along with supporting fields like created_at, lang etc.) from twitter api using `tweepy` library and ingest them into a kafka topic "raw-tweets".

- The script `filter-tweets.py` listens to the Kafka topic "raw-tweets" that writes to the topic “en-tweets” or "fr-tweets" according to tweet language. You can extend to other languages of course.


- The script `sentiment-tweets.py` listens to two Kafka topics "en-tweets" and "fr-tweets" and writes to the topic "positive-tweets" or "negative-tweets" according to the sentiment of tweets. For this, we first cleaned the tweet using `regular expression` then employed `textblob` library to analyse sentiment.

- The script `archive-data.py` archives all topics data (raw-tweets, en-tweets, fr-tweets, positive-tweets, negative-tweets) in text files.

- The script `monitor-kafka.py` monitors all the Kafka topics and prints the status of each Kafka topic in the console in real-time. The status information includes: Topic-name, Partition-id, offset-id, timestamp.

# How to use them?

You need five `cmd` to run respectively those five scripts: ingest-tweets.py, filter-tweets.py, sentiment-tweets.py, archive-data.py and monitor-kafka.py
