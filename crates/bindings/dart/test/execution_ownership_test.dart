import 'dart:async';

import 'package:grafeo/grafeo.dart';
import 'package:test/test.dart';

const _busyQuery =
    'MATCH (a:CancelWork), (b:CancelWork), (c:CancelWork), (d:CancelWork) '
    'WHERE a.i + b.i + c.i + d.i < 0 RETURN a.i';

GrafeoDB _db() => GrafeoDB.memory();

void _seedBusy(GrafeoDB db) {
  db.execute('UNWIND range(1, 128) AS i INSERT (:CancelWork {i: i})');
}

void _expectCode(Object error, String code, GrafeoStatus status) {
  expect(error, isA<GrafeoException>());
  final typed = error as GrafeoException;
  expect(typed.code, code);
  expect(typed.status, status);
}

void main() {
  test('pre-cancel is single-use and does not mutate', () async {
    final db = _db();
    addTearDown(db.close);
    final control = QueryControl();
    control.cancel();

    try {
      db.executeWithOptions(
        'INSERT (:Cancelled {i: 1}) RETURN 1',
        options: ExecutionOptions(control: control),
      );
      fail('cancelled execution succeeded');
    } catch (error) {
      _expectCode(error, 'GRAFEO-Q007', GrafeoStatus.cancelled);
    }
    expect(db.execute('MATCH (n:Cancelled) RETURN n').rows, isEmpty);
    expect(
      () => db.executeWithOptions('RETURN 1',
          options: ExecutionOptions(control: control)),
      throwsA(isA<StateError>()),
    );
    final fresh = QueryControl();
    addTearDown(fresh.close);
    expect(
        db
            .executeWithOptions('RETURN 1',
                options: ExecutionOptions(control: fresh))
            .rows,
        isNotEmpty);
    control.close();
  });

  test('zero timeout is an immediate typed deadline', () {
    final db = _db();
    addTearDown(db.close);
    final control = QueryControl(timeout: Duration.zero);
    addTearDown(control.close);
    expect(
      () => db.executeWithOptions('RETURN 1',
          options: ExecutionOptions(control: control)),
      throwsA(predicate<Object>((error) {
        _expectCode(error, 'GRAFEO-Q003', GrafeoStatus.deadline);
        return true;
      })),
    );
  });

  test('active cancellation interrupts native execution', () async {
    final db = _db();
    addTearDown(db.close);
    _seedBusy(db);
    final control = QueryControl(timeout: const Duration(seconds: 10));
    final future = db.executeWithOptionsAsync(
      _busyQuery,
      options: ExecutionOptions(control: control),
    );
    await Future<void>.delayed(const Duration(milliseconds: 20));
    control.cancel();
    try {
      await future;
      fail('cancelled execution succeeded');
    } catch (error) {
      _expectCode(error, 'GRAFEO-Q007', GrafeoStatus.cancelled);
    } finally {
      control.close();
    }
  });

  test('busy parent close preserves the live database handle', () async {
    final db = _db();
    addTearDown(db.close);
    _seedBusy(db);
    final control = QueryControl(timeout: const Duration(seconds: 10));
    final future = db.executeWithOptionsAsync(
      _busyQuery,
      options: ExecutionOptions(control: control),
    );
    await Future<void>.delayed(const Duration(milliseconds: 20));
    expect(() => db.close(), throwsA(isA<GrafeoException>()));
    control.cancel();
    try {
      await future;
    } catch (_) {
      // Expected cancellation; the close retry below is the ownership check.
    } finally {
      control.close();
    }
    expect(db.execute('RETURN 1').rows, isNotEmpty);
    db.close();
  });

  test('limited transaction statement rolls back while prior write commits',
      () {
    final db = _db();
    addTearDown(db.close);
    final tx = db.beginTransaction();
    tx.execute('INSERT (:Kept {i: 1})');
    try {
      tx.executeWithOptions(
        "INSERT (:Denied {payload: 'large'}) RETURN 1",
        options: ExecutionOptions(maxBytes: 1),
      );
      fail('limited statement succeeded');
    } catch (error) {
      _expectCode(error, 'GRAFEO-S001', GrafeoStatus.resourceLimit);
    }
    tx.commit();
    expect(db.execute('MATCH (n:Kept) RETURN n').rows, hasLength(1));
    expect(db.execute('MATCH (n:Denied) RETURN n').rows, isEmpty);
  });

  test('chunks deliver exactly 2500 unique rows with bounded chunks', () {
    final db = _db();
    addTearDown(db.close);
    final stream =
        db.executeStreamWithOptions('UNWIND range(1, 2500) AS i RETURN i');
    addTearDown(stream.close);
    final seen = <int>{};
    while (true) {
      final chunk = stream.nextChunk(257);
      if (chunk == null) break;
      expect(chunk.rows.length, lessThanOrEqualTo(257));
      for (final row in chunk.rows) {
        expect(seen.add((row['i'] as num).toInt()), isTrue);
      }
    }
    expect(seen, hasLength(2500));
    expect(seen.first, 1);
    expect(seen.last, 2500);
    stream.close();
    stream.close();
  });

  test('collection limit is sticky and does not return partial rows', () {
    final db = _db();
    addTearDown(db.close);
    final stream = db.executeStreamWithOptions(
      'UNWIND [1, 2] AS i RETURN i',
      options: ExecutionOptions(maxRows: 1),
    );
    expect(() => stream.toList(), throwsA(isA<StorageException>()));
    expect(() => stream.next(), throwsA(isA<StorageException>()));
    expect(() => stream.close(), throwsA(isA<StorageException>()));
  });

  test('async iterator early break can close the stream', () async {
    final db = _db();
    addTearDown(db.close);
    final stream =
        db.executeStreamWithOptions('UNWIND range(1, 2500) AS i RETURN i');
    var count = 0;
    await for (final _ in stream.rowsAsync()) {
      count++;
      if (count == 2) break;
    }
    expect(() => stream.next(), throwsA(isA<DatabaseException>()));
    db.close();
    await stream.closeAsync();
    expect(count, 2);
  });
  test('closing consumed control retains worker until its deadline', () async {
    final db = _db();
    addTearDown(db.close);
    _seedBusy(db);
    final control = QueryControl(timeout: const Duration(milliseconds: 100));
    final future = db.executeWithOptionsAsync(_busyQuery,
        options: ExecutionOptions(control: control));
    control.close();
    await expectLater(
        future,
        throwsA(isA<QueryException>()
            .having((e) => e.code, 'code', 'GRAFEO-Q003')));
    expect(db.execute('RETURN 1').rows, hasLength(1));
  });

  test('active stream close cancels and joins before native free', () async {
    final db = _db();
    addTearDown(db.close);
    _seedBusy(db);
    final stream = db.executeStreamWithOptions(_busyQuery);
    final pull = stream.nextAsync();
    final checkedPull = expectLater(
        pull,
        throwsA(isA<QueryException>()
            .having((e) => e.code, 'code', 'GRAFEO-Q007')));
    await Future<void>.delayed(const Duration(milliseconds: 20));
    await expectLater(stream.closeAsync(), throwsA(isA<QueryException>()));
    await checkedPull;
    expect(() => stream.close(), throwsA(isA<QueryException>()));
    db.close();
  });

  test('shape and byte boundaries never deny after mutation commit', () {
    final shapes = [
      '1',
      '{}',
      '[]',
      "{a: [1,2,3], b: {c: 'text'}}",
      "'${List.filled(1000, 'x').join()}'",
      '${List.filled(80, '[').join()}1${List.filled(80, ']').join()}'
    ];
    var admitted = 0;
    var denied = 0;
    for (final shape in shapes) {
      for (final bytes in [256, 1024, 4096, 16384, 65536, 1048576]) {
        final db = _db();
        try {
          try {
            expect(
                db
                    .executeWithOptions(
                        'INSERT (:BudgetProbe) RETURN $shape AS value',
                        options: ExecutionOptions(maxBytes: bytes))
                    .rows,
                hasLength(1));
            expect(db.execute('MATCH (n:BudgetProbe) RETURN n').rows,
                hasLength(1));
            admitted++;
          } on StorageException catch (error) {
            _expectCode(error, 'GRAFEO-S001', GrafeoStatus.resourceLimit);
            expect(db.execute('MATCH (n:BudgetProbe) RETURN n').rows, isEmpty);
            denied++;
          }
        } finally {
          db.close();
        }
      }
    }
    expect(admitted, greaterThan(0));
    expect(denied, greaterThan(0));
  });
  test('transaction worker cancellation preserves earlier writes and parent',
      () async {
    final db = _db();
    addTearDown(db.close);
    _seedBusy(db);
    final tx = db.beginTransaction();
    tx.execute('INSERT (:Earlier)');
    final control = QueryControl();
    addTearDown(control.close);
    final future = tx.executeWithOptionsAsync(_busyQuery,
        options: ExecutionOptions(control: control));
    expect(() => tx.commit(), throwsA(isA<TransactionException>()));
    expect(() => tx.rollback(), throwsA(isA<TransactionException>()));
    expect(() => db.close(), throwsA(isA<GrafeoException>()));
    control.cancel();
    await expectLater(
        future,
        throwsA(isA<QueryException>()
            .having((e) => e.code, 'code', 'GRAFEO-Q007')));
    tx.commit();
    expect(db.execute('MATCH (n:Earlier) RETURN n').rows, hasLength(1));
  });

  test('independent asynchronous streams isolate cancellation', () async {
    final db = _db();
    addTearDown(db.close);
    _seedBusy(db);
    final control = QueryControl();
    addTearDown(control.close);
    final first = db.executeStreamWithOptions(_busyQuery,
        options: ExecutionOptions(control: control));
    final second = db.executeStream('UNWIND range(1, 2500) AS i RETURN i');
    final pull = first.nextAsync();
    final checked = expectLater(
        pull,
        throwsA(isA<QueryException>()
            .having((e) => e.code, 'code', 'GRAFEO-Q007')));
    control.cancel();
    final seen = <int>{};
    await for (final row in second.rowsAsync()) {
      expect(seen.add(row['i'] as int), isTrue);
    }
    await checked;
    expect(seen, hasLength(2500));
    expect(() => first.close(), throwsA(isA<QueryException>()));
  });
  test('temporal marker maps and extreme timestamps cannot fail after commit',
      () {
    final db = _db();
    addTearDown(db.close);
    final payloads = [
      {r'$date': 1},
      {r'$time': false},
      {
        r'$duration': [1]
      },
      {r'$zoned_datetime': 2},
      {r'$timestamp_us': 0x7fffffffffffffff},
      {r'$timestamp_us': -0x8000000000000000},
      {r'$timestamp_us': 'ordinary'},
      {r'$date': 1, 'other': 2},
    ];
    for (final payload in payloads) {
      final result = db.executeWithOptions(
          r'INSERT (:TemporalProbe) RETURN $payload AS value',
          params: {'payload': payload});
      expect(result.rows.single['value'], payload);
    }
    expect(db.execute('MATCH (n:TemporalProbe) RETURN n').rows,
        hasLength(payloads.length));
    for (final microseconds in [-8640000000000000000, 8640000000000000000]) {
      final result = db.executeWithOptions(r'RETURN $value AS value', params: {
        'value': {r'$timestamp_us': microseconds}
      });
      expect(result.rows.single['value'],
          DateTime.fromMicrosecondsSinceEpoch(microseconds, isUtc: true));
    }
  });
}
