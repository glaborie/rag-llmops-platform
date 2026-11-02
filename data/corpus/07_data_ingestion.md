---
doc_id: data_ingestion
title: Streaming Data Ingestion
category: data_ingestion
---

## Clusters, topics, and partitions

Northwind Streaming provides Kafka-compatible clusters. A cluster runs in one region with 3 brokers in the development tier and 9 brokers in the production tier. A topic is split into partitions, the default partition count is 3, and a topic may have at most 500 partitions. The default replication factor is 3 and the default retention is 7 days, after which records are deleted. A tenant may create 1,000 topics per cluster. A single partition sustains 100 MB/s and a cluster sustains 1 million records per second.

## Producers

Producers send records to a topic, and the default settings favour durability over latency. `acks=all` is the default, so the broker waits for the full in-sync replica set to acknowledge a write. Idempotent production is enabled by default, which prevents duplicate records after a producer retry, and the producer identifier survives a reconnect for 5 minutes. The linger time is 5 ms, the batch size is 64 KB, and the compression codec is zstd. A producer may keep 5 in-flight requests per connection.

## Consumers and consumer groups

A consumer belongs to a group, and the partitions of a topic are divided between the members of that group. The session timeout is 45 seconds, and a member that fails to poll within the max poll interval (`max.poll.interval`) of 5 minutes is evicted and its partitions are reassigned. Offsets are committed automatically every 5 seconds, or manually with `enable.auto.commit=false`. A consumer that crashes between processing and committing reprocesses the batch, so the processing must be idempotent. The 5-minute max poll interval is the most common cause of a rebalance loop, and raising it is the first fix to try.

## Schema registry and exactly-once

The schema registry validates records against a schema stored under a subject, named `<topic>-value` for the record payload. Avro, JSON Schema and protobuf are supported. The default compatibility mode is `BACKWARD`, so a new schema may drop optional fields but must not add a required field or change the type of an existing field. Exactly-once delivery is achieved with an idempotent producer plus a transactional id, which commits the read offsets and the produced records atomically. A transactional producer is limited to 60,000 records per second per partition.
