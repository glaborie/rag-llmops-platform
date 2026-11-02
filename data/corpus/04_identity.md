---
doc_id: identity
title: Identity & Access Management
category: identity
---

## Principals, groups, and roles

Northwind IAM manages three principal types. Users represent people and authenticate with a password plus a second factor. Roles represent workloads and are assumed by a user, a service account, or another role. A role carries a trust policy that lists which principals may assume it. Groups simplify user management: a group holds up to 500 members and can be granted the same policies as a user, and a user may belong to 10 groups at once.

## Service accounts and short-lived credentials

A service account is a non-human principal named `nw-sa-<name>` that owns a set of policies and is attached to a compute instance or a job. Instead of a static key, a service account receives an access token that expires after 30 minutes and is rotated automatically every 30 minutes. Callers request a token with `nwctl auth token --service-account <name>` and send it in an `Authorization: Bearer` header. Long-lived API keys are still issued for legacy integrations, but a user may hold at most 2 active keys, and the platform warns after 90 days of use. An assumed role session lasts 1 hour by default and can be extended to a maximum of 12 hours for long-running jobs.

## MFA and password policy

Multi-factor authentication is enforced by default for all console logins. TOTP applications and hardware keys are both accepted, and a second factor is required on any new device. Passwords must be at least 14 characters and a maximum age of 90 days is applied. Administrators may not disable MFA for a user, only for service accounts, which never open a console session. A console session token expires after 12 hours of inactivity.

## Policy structure and permissions

Permissions are written as JSON policy documents with an `Action` list such as `nw:bucket:Read`, a `Resource` such as `nw://acme-data/*`, and an `Effect` of `Allow` or `Deny`. `Deny` always wins. Wildcards are supported in both actions and resources, and the policy lint step flags any grant of `nw:*` on `*`. A principal's effective permissions are the union of its direct grants, its group grants, and the policies attached to the roles it may assume. Every permission change is written to the audit log.
