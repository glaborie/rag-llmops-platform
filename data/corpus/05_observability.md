---
doc_id: observability
title: Observability & Logging
category: observability
---

## Metrics

Northwind Observability collects metrics at three resolutions. Standard metrics are stored at 60-second resolution, fine-grained metrics at 10-second resolution, and high-resolution metrics at 1-second resolution. Standard metrics are free up to 10,000 active series per tenant; beyond that, ingestion is billed per 1,000 series per month. The endpoint `/metrics` on an instance exposes a Prometheus-format scrape target, and the recommended scrape interval is 15 seconds. A tenant can hold at most 5,000 dashboards and 20,000 alert rules.

## Logs and traces

Logs are written to a named log group inside a region. The default retention is 90 days, and the first 30 days act as a hot tier with full-text search at no extra cost; searching older logs takes up to 60 seconds. Traces are sampled at 10% by default, and a request that returns a 5xx is always captured at 100%. Trace data is retained for 30 days. Correlation works by copying the `nw-trace-id` header into the log group, which is the recommended way to pivot from a slow endpoint to the underlying logs.

## Alerting

An alert rule evaluates a metric or log query on a fixed period. The default evaluation period is 5 minutes, that is a 5-minute window, and the default condition requires 2 consecutive periods to breach before the alert fires, which suppresses flapping. A breach that lasts 3 minutes therefore does not page a responder. Notification channels include email, a webhook, and Northwind Oncall for paging. A rule can be silenced for a fixed window, and a silenced rule still evaluates so that its history stays complete. Severity drives routing: `critical` alerts page, `warning` alerts send email.

## Retention, cost, and access

Metrics are retained for 15 months. Logs are retained for 90 days unless a longer retention is configured on the log group, and traces are retained for 30 days. The observability service emits its own usage metrics, so watching the cost of telemetry is the same exercise as watching an application. Access to dashboards and log groups is granted with IAM roles, and the short-lived credentials described in the identity guide are used, so read access to a log group and write access to a billing account are both expressed as role policies.
