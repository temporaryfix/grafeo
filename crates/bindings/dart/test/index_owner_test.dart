import 'dart:ffi';
import 'dart:io';

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

void main() {
  late GrafeoDB db;
  setUp(() => db = GrafeoDB.memory());
  tearDown(() => db.close());

  test('owner lifecycle distinguishes absence from failure', () {
    final owner = db.createIndex(
        const CreateIndexRequest(kind: IndexKind.property, property: 'name'));
    db.rebuildIndex(owner);
    expect(db.dropIndex(owner), isTrue);
    expect(db.dropIndex(owner), isFalse);
    expect(() => db.rebuildIndex(owner), throwsException);
  });

  test('component-qualified requests do not alias root or truncate NUL', () {
    db.execute('CREATE GRAPH scoped');
    final root = db.createIndex(
        const CreateIndexRequest(kind: IndexKind.property, property: 'name'));
    final scoped = db.createIndex(const CreateIndexRequest(
        kind: IndexKind.property, property: 'name', graph: ['scoped']));
    expect(root, isNot(scoped));
    expect(db.dropIndex(scoped), isTrue);
    db.rebuildIndex(root);
    expect(
        () => db.createIndex(const CreateIndexRequest(
            kind: IndexKind.property,
            property: 'unique',
            graph: ['scoped\u0000other'])),
        throwsException);
    expect(
        () => db.createIndex(const CreateIndexRequest(
            kind: IndexKind.property, property: 'bad', dimensions: 0)),
        throwsException);
  });

  test('every index request string rejects malformed UTF-16', () {
    for (final malformed in [
      String.fromCharCode(0xd800),
      String.fromCharCode(0xdc00),
      String.fromCharCodes([0xd800, 0x61]),
    ]) {
      final requests = [
        CreateIndexRequest(kind: IndexKind.property, property: malformed),
        CreateIndexRequest(
            kind: IndexKind.property, property: 'p', name: malformed),
        CreateIndexRequest(
            kind: IndexKind.property, property: 'p', graph: [malformed]),
        CreateIndexRequest(
            kind: IndexKind.vector,
            property: 'p',
            label: malformed,
            dimensions: 3),
        CreateIndexRequest(
            kind: IndexKind.vector,
            property: 'p',
            label: 'Doc',
            dimensions: 3,
            metric: malformed),
        CreateIndexRequest(
            kind: IndexKind.vector,
            property: 'p',
            label: 'Doc',
            dimensions: 3,
            quantization: malformed),
      ];
      for (final request in requests) {
        expect(() => db.createIndex(request), throwsArgumentError);
      }
    }
  });

  test('valid surrogate pairs retain their exact index identity', () {
    final supplementary = String.fromCharCodes([0xd83d, 0xde00]);
    final owner = db.createIndex(CreateIndexRequest(
      kind: IndexKind.property,
      property: supplementary,
      name: 'index_$supplementary',
    ));
    expect(db.hasPropertyIndex(supplementary), isTrue);
    expect(db.hasPropertyIndex('\ufffd'), isFalse);
    expect(db.dropIndex(owner), isTrue);
  });

  test('vector creation and rebuild use one owner', () {
    db.execute('INSERT (:Doc {embedding: [1.0, 0.0, 0.0]})');
    final owner = db.createIndex(const CreateIndexRequest(
        kind: IndexKind.vector,
        label: 'Doc',
        property: 'embedding',
        dimensions: 3));
    db.rebuildIndex(owner);
    expect(db.vectorSearch('Doc', 'embedding', [1.0, 0.0, 0.0], k: 1),
        hasLength(1));
    expect(db.dropIndex(owner), isTrue);
  });

  test('text tokenizer presence survives rebuild and reopen', () {
    db.execute(
      "INSERT (:Doc {body: 'x ox fox lengthy', default_body: 'x ox fox lengthy', zero_body: 'x ox fox lengthy'})",
    );
    const request = CreateIndexRequest(
      kind: IndexKind.text,
      label: 'Doc',
      property: 'body',
      minTokenLength: 7,
    );
    final owner = db.createIndex(request);
    for (final kind in [
      IndexKind.property,
      IndexKind.btree,
      IndexKind.vector,
    ]) {
      expect(
        () => db.createIndex(
          CreateIndexRequest(
            kind: kind,
            property: 'bad',
            minTokenLength: 0,
            label: kind == IndexKind.vector ? 'Doc' : null,
            dimensions: kind == IndexKind.vector ? 3 : null,
          ),
        ),
        throwsException,
      );
    }
    for (final minimum in [-1, if (sizeOf<IntPtr>() == 4) 0x100000000]) {
      expect(
        () => db.createIndex(
          CreateIndexRequest(
            kind: IndexKind.text,
            label: 'Doc',
            property: 'bad',
            minTokenLength: minimum,
          ),
        ),
        throwsRangeError,
      );
    }
    expect(() => db.createIndex(request), throwsException);
    final defaultOwner = db.createIndex(
      const CreateIndexRequest(
        kind: IndexKind.text,
        label: 'Doc',
        property: 'default_body',
      ),
    );
    expect(
      defaultOwner,
      owner + 1,
      reason: 'rejected requests must not consume owner IDs',
    );
    final zeroOwner = db.createIndex(
      const CreateIndexRequest(
        kind: IndexKind.text,
        label: 'Doc',
        property: 'zero_body',
        minTokenLength: 0,
      ),
    );
    void check(GrafeoDB candidate) {
      for (final (property, token, count) in [
        ('body', 'fox', 0),
        ('body', 'lengthy', 1),
        ('default_body', 'x', 0),
        ('default_body', 'ox', 1),
        ('zero_body', 'x', 1),
      ]) {
        expect(
          candidate
              .execute(
                "CALL grafeo.search.text('Doc', '$property', '$token', 10)",
              )
              .rows,
          hasLength(count),
          reason: '$property/$token',
        );
      }
    }

    check(db);
    for (final id in [owner, defaultOwner, zeroOwner]) {
      db.rebuildIndex(id);
    }
    check(db);
    final directory = Directory.systemTemp.createTempSync('grafeo-dart-text-');
    try {
      final path =
          '${directory.path}${Platform.pathSeparator}text-options.grafeo';
      db.save(path);
      final reopened = GrafeoDB.open(path);
      try {
        check(reopened);
        for (final id in [owner, defaultOwner, zeroOwner]) {
          reopened.rebuildIndex(id);
        }
        check(reopened);
      } finally {
        reopened.close();
      }
    } finally {
      directory.deleteSync(recursive: true);
    }
  });
}
