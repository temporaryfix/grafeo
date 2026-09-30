# Review scope

This branch is an implementation proposal for maintainer review. It does not
change the upstream roadmap or assign features to a release.

The proposed review covers transactions and recovery across LPG/RDF, retained
history, exact bounded query execution, query and path correctness, backup and
durable feeds, and the associated bindings.

The upstream [milestones](https://github.com/GrafeoDB/grafeo/milestones) and
[project board](https://github.com/orgs/GrafeoDB/projects/1) define release scope.
Compatibility is the starting point for contributions. Existing API and format
differences in this development snapshot require individual review, not a
blanket compatibility break. See the root `REVIEW.md` for the current upstream
fit, implementation sections and validation limits.
