---
doc_id: api_reference
title: API Reference
category: api_reference
---

## Base URL, versioning, and content type

The Northwind API is a REST API. The base URL is `https://api.northwind.example/v1`, and the major version sits in the path, so `/v1` and `/v2` can run side by side. Request and response bodies are JSON, and the `Content-Type: application/json` header is required on every request that carries a body. Resource names are plural and hyphenated, such as `/v1/object-buckets` and `/v1/compute-instances`. Field names are `snake_case`, and timestamps are UTC in RFC 3339 format, for example `2026-03-14T09:15:00Z`.

## Authentication

Two authentication methods are supported. Interactive clients send `Authorization: Bearer <token>` with a token from `nwctl auth token`. Machine clients sign the request with `X-Northwind-Key` and `X-Northwind-Signature`, where the signature is an HMAC-SHA256 of the method, path and body using the key secret, and the timestamp travels in `X-Northwind-Timestamp`. A request whose timestamp is more than 5 minutes away from the server clock is rejected with `NW-2201`. Static API keys are accepted for legacy integrations and are limited to 2 active keys per user.

## Pagination

List endpoints are cursor-paginated. Send `limit` to choose the page size, and the default is 25 with a maximum of 200. The response contains a `data` array plus a `next_cursor` field, and `next_cursor` is `null` on the final page. A `cursor` query parameter continues from a previous page, and a cursor stays valid for 15 minutes. Cursors are opaque strings and must not be parsed. Filtering uses query parameters such as `region=` and `status=`, and several filters are combined with AND.

## Errors, rate limits, and idempotency

An error response is a JSON object with an `error` object that holds `code`, `message` and `request_id`. The HTTP status is 400 for a malformed request, 401 for an invalid or expired token, 403 for an insufficient permission, 404 for a missing resource, 409 for a conflict such as a name already in use, 422 for a schema validation failure, 429 for a rate limit, and 5xx for a server fault. Every response carries an `X-Request-Id` header that matches the `request_id` in an error body, and that identifier is the first thing to quote in a support ticket.

The rate limit is 600 requests per minute per tenant, evaluated over a sliding one-minute window. Responses include `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset`, and a 429 response also includes a `Retry-After` header in seconds. A request that would exceed the limit is rejected rather than queued, and the retry guidance lives in the troubleshooting runbook.

A POST request that creates a resource accepts an `Idempotency-Key` header. The key must be unique per logical operation and is stored for 24 hours. A repeat of the same key inside that window returns the original response with the same status code and does not create a second resource. Keys are scoped to the tenant.
