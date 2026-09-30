import 'dart:io';
import 'dart:typed_data';

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

void main() {
  final fixture = Platform.environment['GRAFEO_CDC_EVICTED_FIXTURE'];
  test(
    'retained cut preserves stale cursor in every page reader',
    () {
      final cursor = File(
        fixture!.replaceFirst(RegExp(r'\.grafeo$'), '.cursor'),
      ).readAsBytesSync();
      expect(cursor.length, 97);
      final dir = Directory.systemTemp.createTempSync('grafeo-retained-dart-');
      addTearDown(() => dir.deleteSync(recursive: true));
      final path = '${dir.path}/retained.grafeo';
      File(fixture).copySync(path);
      final db = GrafeoDB.open(path);
      try {
        for (final read in <ChangePage Function()>[
          () => db.changesAfter(cursor, maxEvents: 1, maxBytes: 4096),
          () => db.nodeHistoryAfter(
            BigInt.zero,
            cursor,
            maxEvents: 1,
            maxBytes: 4096,
          ),
          () => db.edgeHistoryAfter(
            BigInt.zero,
            cursor,
            maxEvents: 1,
            maxBytes: 4096,
          ),
        ]) {
          expect(
            read,
            throwsA(
              isA<GrafeoException>().having(
                (e) => e.code,
                'code',
                'GRAFEO-S006',
              ),
            ),
          );
        }
        expect(
          db.changesAfter(null, maxEvents: 1, maxBytes: 4096).events,
          hasLength(1),
        );
      } finally {
        db.close();
      }
    },
    skip: fixture == null
        ? 'generate the native C retained-cut fixture first'
        : false,
  );

  test('owned pages retain creation payload and bounded entity history', () {
    final db = GrafeoDB.memory();
    addTearDown(db.close);
    db.cdcEnabled = true;
    expect(db.cdcEnabled, isTrue);
    final a = db.createNode(['N'], {'large': 9007199254740993});
    final b = db.createNode(['N'], {});
    final edge = db.createEdge(a, b, 'LINK', {});
    final pages = <ChangePage>[];
    Uint8List? cursor;
    for (final id in [a, b, edge]) {
      final page = db.changesAfter(cursor, maxEvents: 1, maxBytes: 4096);
      expect(page.events.single.entityId, BigInt.from(id));
      expect(page.next.length, 97);
      pages.add(page);
      cursor = page.next;
    }
    final eof = db.changesAfter(cursor, maxEvents: 1, maxBytes: 4096);
    expect(eof.events, isEmpty);
    expect(eof.next, cursor);
    final node = db
        .nodeHistoryAfter(BigInt.from(a), null, maxEvents: 1, maxBytes: 4096)
        .events
        .single;
    expect(
      db
          .nodeHistoryAfter(
            BigInt.from(a),
            null,
            sinceEpoch: node.epoch,
            maxEvents: 1,
            maxBytes: 4096,
          )
          .events
          .single
          .epoch,
      node.epoch,
    );
    expect(
      db
          .nodeHistoryAfter(
            BigInt.from(a),
            null,
            sinceEpoch: node.epoch + BigInt.one,
            maxEvents: 1,
            maxBytes: 4096,
          )
          .events,
      isEmpty,
    );
    expect(
      db
          .edgeHistoryAfter(
            BigInt.from(edge),
            null,
            maxEvents: 1,
            maxBytes: 4096,
          )
          .events
          .single
          .entityType,
      'edge',
    );
    expect(
      db
          .nodeHistoryAfter(
            BigInt.parse('18446744073709551614'),
            null,
            maxEvents: 1,
            maxBytes: 4096,
          )
          .events,
      isEmpty,
    );
    expect(
      db
          .nodeHistoryAfter(
            BigInt.from(a),
            null,
            sinceEpoch: (BigInt.one << 64) - BigInt.one,
            maxEvents: 1,
            maxBytes: 4096,
          )
          .events,
      isEmpty,
    );
    for (final bad in [-BigInt.one, BigInt.one << 64]) {
      expect(
        () => db.nodeHistoryAfter(bad, null, maxEvents: 1, maxBytes: 4096),
        throwsArgumentError,
      );
    }
    db.close();
    expect(pages[0].events.single.labels, ['N']);
    expect(pages[0].events.single.after!['large'], 9007199254740993);
    expect(pages[0].events.single.graphIncarnation, isNotNull);
    expect(pages[0].events.single.lpgGraph, isEmpty);
    expect(pages[2].events.single.edgeType, 'LINK');
    expect(pages[2].events.single.sourceId, BigInt.from(a));
    expect(pages[2].events.single.targetId, BigInt.from(b));
    expect(
      () => db.changesAfter(null, maxEvents: 1, maxBytes: 4096),
      throwsA(isA<DatabaseException>()),
    );
  });

  test('invalid bounds and cursors preserve native error codes', () {
    final db = GrafeoDB.memory();
    addTearDown(db.close);
    db.cdcEnabled = true;
    db.createNode(['N'], {});
    Matcher code(String value) =>
        throwsA(isA<GrafeoException>().having((e) => e.code, 'code', value));
    for (final length in [0, 96, 97, 98]) {
      expect(
        () => db.changesAfter(Uint8List(length), maxEvents: 1, maxBytes: 4096),
        code('GRAFEO-S004'),
      );
    }
    for (final bounds in [
      [0, 4096],
      [-1, 4096],
      [1, 0],
      [1, -1],
    ]) {
      expect(
        () => db.changesAfter(null, maxEvents: bounds[0], maxBytes: bounds[1]),
        code('GRAFEO-V001'),
      );
    }
    expect(
      () => db.changesAfter(null, maxEvents: 1, maxBytes: 1),
      code('GRAFEO-S001'),
    );
    final other = GrafeoDB.memory();
    addTearDown(other.close);
    other.cdcEnabled = true;
    final foreign = other.changesAfter(null, maxEvents: 1, maxBytes: 4096).next;
    expect(
      () => db.changesAfter(foreign, maxEvents: 1, maxBytes: 4096),
      code('GRAFEO-S005'),
    );
    expect(
      db.changesAfter(null, maxEvents: 1, maxBytes: 4096).events,
      hasLength(1),
    );
  });

  test('page one resumes across two durable reopens', () {
    final dir = Directory.systemTemp.createTempSync('grafeo-cdc-dart-');
    addTearDown(() => dir.deleteSync(recursive: true));
    final path = '${dir.path}/store';
    final db = GrafeoDB.open(path);
    db.cdcEnabled = true;
    final ids = List.generate(3, (_) => db.createNode(['N'], {}));
    final first = db.changesAfter(null, maxEvents: 1, maxBytes: 4096);
    expect(first.events.single.entityId, BigInt.from(ids[0]));
    var cursor = first.next;
    db.close();
    for (var i = 1; i < 3; i++) {
      final reopened = GrafeoDB.open(path);
      try {
        final page = reopened.changesAfter(
          cursor,
          maxEvents: 1,
          maxBytes: 4096,
        );
        expect(page.events.single.entityId, BigInt.from(ids[i]));
        cursor = page.next;
        if (i == 2) {
          final eof = reopened.changesAfter(
            cursor,
            maxEvents: 1,
            maxBytes: 4096,
          );
          expect(eof.events, isEmpty);
          expect(eof.next, cursor);
        }
      } finally {
        reopened.close();
      }
    }
  });

  test('JSON retains unsigned maximum and native RDF fields', () {
    final ev = ChangeEvent.fromJson({
      'entity_id': '18446744073709551615',
      'epoch': '18446744073709551614',
      'timestamp': '18446744073709551613',
      'graph_incarnation': '18446744073709551612',
      'src_id': '18446744073709551611',
      'dst_id': '18446744073709551610',
      'triple_graph': '<urn:g>',
      'triple_subject': '<urn:s>',
      'triple_predicate': '<urn:p>',
      'triple_object': '"value"',
    });
    final maximum = (BigInt.one << 64) - BigInt.one;
    expect(ev.entityId, maximum);
    expect(ev.epoch, maximum - BigInt.one);
    expect(ev.timestamp, maximum - BigInt.two);
    expect(ev.graphIncarnation, maximum - BigInt.from(3));
    expect(ev.sourceId, maximum - BigInt.from(4));
    expect(ev.targetId, maximum - BigInt.from(5));
    expect(ev.tripleGraph, '<urn:g>');
    expect(ev.tripleSubject, '<urn:s>');
    expect(ev.triplePredicate, '<urn:p>');
    expect(ev.tripleObject, '"value"');
  });
}
