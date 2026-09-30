---
title: Grafeo implementation review
description: Development source for maintainer review; not a released version.
---

# Grafeo implementation review

This source extends Grafeo's embedded property-graph and RDF engine with retained
history, shared transaction and recovery contracts, bounded query execution and
durable change feeds. It is an unreleased review candidate.

Start with [native host profiles](user-guide/native-host.md),
[retained history](user-guide/temporal.md), and
[transactions](user-guide/transactions.md). The root `REVIEW.md` describes the
review sections, evidence and outstanding qualification.

Historical visibility is distinct from retention. Unsupported conformance cases
remain explicit, and platform support must be established by the relevant tests.
The inherited documentation is not a claim of release readiness.
