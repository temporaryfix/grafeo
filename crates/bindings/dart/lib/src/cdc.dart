/// Owned bounded native change pages.
library;

import 'dart:convert';
import 'dart:typed_data';

/// One committed event. BigInt coordinates preserve every bit of native u64.
/// Creation/RDF payloads come from the commit, not a later graph lookup.
class ChangeEvent {
  final Map<String, dynamic> _json;
  ChangeEvent.fromJson(Map<String, dynamic> json) : _json = Map.of(json);

  BigInt get entityId => BigInt.parse(_json['entity_id'] as String);
  String get entityType => _json['entity_type'] as String;
  String get kind => _json['kind'] as String;
  BigInt get epoch => BigInt.parse(_json['epoch'] as String);
  BigInt get timestamp => BigInt.parse(_json['timestamp'] as String);
  BigInt? _coordinate(String key) =>
      _json[key] == null ? null : BigInt.parse(_json[key] as String);
  BigInt? get graphIncarnation => _coordinate('graph_incarnation');
  BigInt? get sourceId => _coordinate('src_id');
  BigInt? get targetId => _coordinate('dst_id');
  Map<String, dynamic>? get before =>
      (_json['before'] as Map?)?.cast<String, dynamic>();
  Map<String, dynamic>? get after =>
      (_json['after'] as Map?)?.cast<String, dynamic>();
  List<String>? get labels => (_json['labels'] as List?)?.cast<String>();
  String? get edgeType => _json['edge_type'] as String?;
  List<String>? get lpgGraph => (_json['lpg_graph'] as List?)?.cast<String>();
  String? get tripleGraph => _json['triple_graph'] as String?;
  String? get tripleSubject => _json['triple_subject'] as String?;
  String? get triplePredicate => _json['triple_predicate'] as String?;
  String? get tripleObject => _json['triple_object'] as String?;
}

/// Managed events and exclusive cursor; stopping early needs no native cleanup.
/// Empty filtered pages may advance the cursor. Unchanged next means EOF.
class ChangePage {
  final List<ChangeEvent> events;
  final Uint8List next;

  ChangePage.fromJson(String json, Uint8List cursor)
    : events = (jsonDecode(json) as List)
          .map(
            (value) =>
                ChangeEvent.fromJson((value as Map).cast<String, dynamic>()),
          )
          .toList(growable: false),
      next = Uint8List.fromList(cursor);
}
