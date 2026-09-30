/// Canonical index families. Disabled families return an engine error.
enum IndexKind { property, btree, text, vector }

/// One graph-qualified index creation request.
///
/// An empty [graph] selects root; each string is one literal component, including
/// empty strings, separators and embedded NULs. Null options are absent; a
/// supplied zero or empty string is passed to engine validation unchanged.
final class CreateIndexRequest {
  final List<String> graph;
  final String? name;
  final String? label;
  final String property;
  final IndexKind kind;
  final int? dimensions;
  final String? metric;
  final int? m;
  final int? efConstruction;

  /// Text-only; null selects the default (2), while zero is valid.
  final int? minTokenLength;
  final String? quantization;

  const CreateIndexRequest({
    this.graph = const [],
    this.name,
    this.label,
    required this.property,
    required this.kind,
    this.dimensions,
    this.metric,
    this.m,
    this.efConstruction,
    this.minTokenLength,
    this.quantization,
  });
}
