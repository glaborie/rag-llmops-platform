---
doc_id: storage
title: Object Storage
category: storage
---

## Bucket structure and naming

Northwind Object Storage keeps data in buckets that belong to a single region. Bucket names are 3 to 63 characters long, use lowercase letters, digits and hyphens, and must be globally unique. Objects are addressed as `nw://<bucket>/<key>`, where the key is an opaque string of up to 1,024 bytes and may contain forward slashes. Each bucket is reached through a regional endpoint such as `https://obj.eu-west-1.northwind.example`.

## Lifecycle policies and retention

Buckets do not keep objects forever. A bucket with no explicit lifecycle policy applies the platform default, which retains objects for 90 days and then expires them. An expired object is unreadable and bills at zero, but recovery is possible only for 7 days afterwards through the version restore path. Lifecycle rules are evaluated daily at 02:00 UTC. A common starting rule transitions objects to `nw-cold` after 30 days and expires them at 90 days. Object locking adds write-once-read-many retention of up to 7 years for compliance workloads.

## Storage classes and replication

Three storage classes are available. `nw-standard` is hot storage with millisecond access and no minimum duration. `nw-cold` costs roughly 40% less and suits data read less than once a month. `nw-archive` is the cheapest tier, with a 12-hour restore latency and a 90-day minimum storage duration. Cross-region replication runs asynchronously and normally places a write in the replica within 5 minutes, which is the replication target the platform holds for every standard bucket. Bucket versioning keeps old generations for 30 days by default.

## Signed URLs and access

Private objects are served through pre-signed URLs that embed an expiry timestamp. The maximum expiry for a signed URL is 7 days, and a request beyond that is rejected. A signed URL grants access to one object and one key; a URL signed for PUT also permits overwriting that object, so treat it as a temporary credential. Every use of a signed URL is recorded in the audit log with the requester identity and the object key, which is the same log reviewed during incident triage.
