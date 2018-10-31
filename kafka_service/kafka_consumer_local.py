#!/usr/bin/env python
# coding: UTF-8

from pykafka import KafkaClient

host = "localhost"
client = KafkaClient(hosts="%s:9092"%host)

print(client.topics)

# Consumer
topic = client.topics['test']
consumer = topic.get_simple_consumer()

for message in consumer:
    if message is not None:
        print(message.offset, message.value)