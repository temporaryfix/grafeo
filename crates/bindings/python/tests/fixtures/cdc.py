"""Finite CDC fixture collection through required bounded public pages."""


def collect_history(db, entity_id, *, edge=False, since_epoch=0):
    reader = db.edge_history_after if edge else db.node_history_after
    cursor = None
    events = []
    while True:
        page = reader(entity_id, cursor, 1, 1024 * 1024, since_epoch=since_epoch)
        assert len(page["events"]) <= 1
        if page["next"] == cursor:
            return events
        cursor = page["next"]
        events.extend(page["events"])
