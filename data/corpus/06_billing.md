---
doc_id: billing
title: Billing & Cost Management
category: billing
---

## Invoices and payment terms

Billing is monthly and usage-based. The statement period runs from the 1st to the last day of the month, and the invoice is issued between the 1st and the 5th of the following month. Payment terms are net 30 days from the invoice date. An invoice lists charges by service, by region, and by tag, and tax is calculated per jurisdiction as a separate line. Usage continues to accrue while a payment is outstanding, and an account more than 60 days overdue has its resources suspended after a 7-day notice. Metered usage is reported with a 24-hour delay, so the final figure can move slightly after the invoice is issued.

## Quotas

A new tenant starts on the standard quota tier. The default limit is 50 vCPUs and 20 running compute instances per region. Object storage is limited to 10 TB per bucket and 50 TB per tenant, a tenant may create 1,000 streaming topics per cluster and 200 VPCs, and each API tenant is limited to 600 requests per minute. A quota increase is requested from support and is usually approved within one business day when the account has no sustained quota errors. When a limit is reached the API returns the error code `NW-4091` rather than queueing the request.

## Budget alerts and cost allocation

A budget is defined for a tenant, a project, or a tag, with an optional monthly ceiling in dollars. Alerts fire at 50%, 80%, and 100% of the budget, and the 100% alert is a notification only; the platform never shuts down a workload automatically. Notifications go to the email addresses on the budget and to an optional webhook. Cost allocation depends on the `cost_center`, `project`, and `env` tags. Untagged spend is reported in a separate unallocated bucket rather than spread across projects, and a chargeback report for a month is available in CSV after the invoice is issued.

## Reservations and credits

Reserved capacity is sold in 1-year and 3-year terms. A 1-year reservation gives a discount of up to 30% against the on-demand rate, and a 3-year reservation gives up to 50%. A reservation covers one family and size in one region and applies automatically to matching on-demand instances. New tenants receive a $25 monthly credit for the first three months, and a promotional credit is applied before the outstanding balance is charged. Spending against a reservation appears on the same invoice line as on-demand usage.
