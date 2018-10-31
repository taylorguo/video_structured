#!/usr/bin/env python
# coding: UTF-8
# Python3.6

from pykafka import KafkaClient
import time

class KafkaConsumer():

    def __init__(self, hosts, topic):
        self.hosts =  hosts
        # self.zookeeper = zookeeper
        self.topic = topic
        # self.consumer_group = consumer_group

    def consumer_message(self):
        dtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        client = KafkaClient(hosts=self.hosts)
        print(client.topics)
        if self.topic in client.topics.keys():
            topic = client.topics[self.topic]
            consumer = topic.get_balanced_consumer(
                # consumer_group=self.consumer_group,
                # zookeeper_connect=self.zookeeper
            )
            consumer.consume()
            consumer.commit_offsets()
            for message in consumer:
                print(type(message))
                print(dtime, message.offset, message.value)

            return True


def main():
    client = KafkaConsumer("192.168.77.3:9092,192.168.77.4:9092,192.168.77.5:9092", "tethys-request")
    client.consumer_message()


if __name__ == "__main__":
    main()