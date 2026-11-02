---
doc_id: troubleshooting
title: Troubleshooting Runbook
category: troubleshooting
---

## Diagnostic commands

`nwctl doctor` is the first command to run for any connectivity or configuration problem. It checks the local configuration file, the certificate, the token expiry, the DNS resolution of the endpoint, and the reachability of the regional API, and it prints a pass or fail line for every check. `nwctl logs tail --group <name> --since 15m` streams the last minutes of a log group, and `nwctl describe <resource>` shows the current state of a single resource. All three commands support `--output json`, and that output attaches directly to a support ticket. The diagnostics are read-only; nothing in `nwctl doctor` changes a deployed resource.

## Common error codes

`NW-2201` means the request carried an invalid or expired credential, and the fix is to refresh the token or to correct the HMAC timestamp. `NW-4032` means the principal lacks the required permission, and the response names the missing action, such as `nw:bucket:Read`. `NW-4291` means the tenant exceeded the rate limit. `NW-4091` means a quota was reached, and the response names the quota. `NW-5021` is an upstream service fault on the Northwind side, and the response carries an `X-Request-Id`. A 502 or a 503 response is safe to retry, while a 400 or a 422 is not, because the request itself has to change.

## Retry and recovery

Retry a 429, a 502 or a 503 with exponential backoff and full jitter, starting at 1 second and doubling, and make no more than 3 attempts. A 429 also carries `Retry-After`, which overrides the computed delay. Do not retry a 409 without reconciling the resource state first, because the original request may have partially succeeded. A client that must not lose work sends an `Idempotency-Key` on every POST, and the API reference states how long a key is remembered. The service status page at `status.northwind.example` shows live incidents, and a spike of `NW-5021` errors that is not listed there is usually a customer-side firewall or DNS issue.

## Escalation and severity

Four severities are available. `P1` is a full outage with no workaround, `P2` is a major degradation or a partial outage, `P3` is a limited feature failure that has a workaround, and `P4` is a general question or a cosmetic defect. A `P1` ticket receives a first human response within 15 minutes, a `P2` within 1 hour, a `P3` within 4 hours, and a `P4` within 1 business day. Open the ticket from the console, attach the `nwctl doctor --output json` output and the `X-Request-Id`, and include the region, the affected resource ID and the first timestamp of the error. Escalate a `P2` to `P1` in the same thread when the workaround stops working.
