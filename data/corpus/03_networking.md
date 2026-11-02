---
doc_id: networking
title: Networking & VPC
category: networking
---

## VPC and subnets

A VPC is scoped to one region and is created with a CIDR block; the recommended size is a /16, for example `10.40.0.0/16`, which leaves room for a /24 subnet in every availability zone. Northwind regions expose three availability zones. A VPC spans one or more subnets, and a subnet covers a single availability zone. Public subnets carry a public address range and route to an internet gateway, while private subnets route outbound through a NAT gateway. One NAT gateway per availability zone is recommended, and each NAT gateway sustains up to 20 Gbps.

## Security groups

Security groups are stateful, so the response to an allowed inbound connection is automatically allowed outbound. A new group applies a default deny to all inbound traffic and allows all outbound traffic by default, so an unmatched inbound connection is dropped. Each group accepts up to 100 rules, and a VPC may hold 500 groups. A rule may reference another security group as its source, which keeps policies readable. Cross-VPC peering requires explicit rules on both sides of the peering.

## Load balancers

Three load balancer types are available. `nw-lb` is a layer 4 balancer for TCP and UDP, `nw-alb` is a layer 7 balancer that understands HTTP headers and paths, and `nw-nlb` is a network balancer tuned for very high connection counts. Health checks run every 10 seconds, a target is marked unhealthy after 3 consecutive failures, and the request timeout is 5 seconds. Registering a target takes effect within 10 seconds, and deregistration drains connections for 300 seconds before the target is removed.

## Egress and private connectivity

Traffic that leaves Northwind is billed. Internet egress costs $0.09 per GB, and the first 100 GB per month is included at no charge. Traffic between availability zones of the same region costs $0.01 per GB, while traffic inside a single availability zone is free. Storage and streaming endpoints are reachable from a VPC through a private endpoint, which keeps object access traffic off the metered egress path and off the public internet.
