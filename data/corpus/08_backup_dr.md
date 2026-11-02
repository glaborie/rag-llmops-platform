---
doc_id: backup_dr
title: Backup & Disaster Recovery
category: backup_dr
---

## Snapshots and backup schedules

Every block volume supports point-in-time snapshots that are incremental. The daily snapshot runs at 02:00 UTC and is kept for 30 days, the weekly snapshot is kept for 12 weeks, and the monthly snapshot is kept for 7 years. A snapshot of a 500 GB volume transfers only the changed blocks. A volume may hold at most 1,000 snapshots, and the oldest automatic snapshot is deleted first when that limit is reached. A snapshot of the root volume of a running instance is crash-consistent, and an application-level checkpoint is required for a clean application state.

## Recovery objectives

The platform publishes two recovery objectives. The RPO for a streaming topic is 5 minutes, and the RPO for a block volume is 15 minutes. The RTO for a regional failover is 30 minutes, while the RTO for restoring a single volume from a snapshot is 60 minutes. Backup retention is set so that a restore is always possible inside the RTO, which makes the daily snapshot the minimum restore point and the replication target the maximum. A restore runs in the target region and the source volume is preserved for 7 days afterwards so that the restore can be rolled back.

## Cross-region replication

Cross-region replication copies snapshots and streaming topics to a paired region asynchronously. The replication lag target is 5 minutes for streaming topics and 15 minutes for volumes, and the state of each relationship is visible in the console and through `nwctl replication status`. A tenant may have 10 active cross-region relationships. A relationship can be paused, which holds the source in place, or suspended, which leaves the replica behind. The replica region must sit on the same continent.

## Failover

Regional failover is a manual promotion of the standby resources, not an automatic DNS switch. A failover promotion takes 10 minutes to run and completes inside the 30-minute RTO, validation traffic included. `nw-dns` health checks run every 30 seconds and route traffic after 3 consecutive failures, with a 60-second record TTL. Before promoting, confirm that the replica lag is under 5 minutes and take a manual snapshot of the standby. Failback is a second promotion in the opposite direction, it is not automated, and it should be scheduled outside business hours.
