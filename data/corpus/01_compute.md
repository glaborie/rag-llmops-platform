---
doc_id: compute
title: Compute Instances
category: compute
---

## Instance families and sizing

Northwind Compute Instances are grouped into four families. `nw-standard` is the general-purpose family with a 1:4 vCPU-to-memory ratio and is the default choice for new workloads. `nw-memory` provides a 1:8 ratio for in-memory caches and analytics. `nw-gpu` exposes 1, 4, or 8 accelerator cores for training and inference. `nw-compute` is a burstable family with a 1:2 ratio that bills only while the instance is actively running.

Instance sizes are named `<family>.<size>`. `s` is 2 vCPU and 8 GB RAM, `m` is 4 and 16, `l` is 8 and 32, `xl` is 16 and 64. `nw-gpu.m` carries 4 accelerator cores with 32 GB of host memory, while `nw-gpu.l` carries 8 cores with 64 GB. Changing the family of a running instance requires a stop and start; changing only the size requires a restart.

## Root volumes and metadata

Every instance boots from a block volume. The default root volume is 50 GB of `nw-block` storage and can be extended online to 16 TB. The root volume is destroyed together with the instance unless you detach it first, so production data must live on a separate volume. `nwctl describe instance` prints the instance ID, the family and size, the availability zone, the attached volume IDs, and the tags applied at launch. Tags such as `cost_center` and `project` are what cost allocation later groups by, and the same tagging rules are used when a workload is moved to a spot capacity pool.

## Autoscaling

A `nw-autoscaler` group holds a minimum and a maximum instance count plus a target utilisation metric. The default target is 65% average CPU over a 5-minute window, with a minimum of 2 and a maximum of 20 instances. The scale-out cooldown is 120 seconds and the scale-in cooldown is 300 seconds, which keeps a bursty workload from flapping. Scaling groups are regional, and new instances are placed in subnets tagged `autoscaling: enabled`.

## Spot capacity and interruption

Spot instances are drawn from spare capacity at a 70% discount against the on-demand rate. Northwind issues a 2-minute interruption notice before reclaiming a spot instance, delivered as an `NW_SPOT_INTERRUPTION` event and written to the instance log group. Workloads that must survive interruption should checkpoint to the block volume at least every 60 seconds and should be spread over at least three availability zones. A spot instance cannot be the only member of a production load balancer target pool.
