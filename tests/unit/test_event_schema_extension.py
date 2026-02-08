"""Tests for event schema extension (pass + set-piece metadata)."""

from src.events.detection import EVENT_METADATA_SCHEMA_VERSION, Event


def test_pass_event_metadata_is_schema_versioned() -> None:
    """Pass events should include schema/family defaults."""
    event = Event(
        event_type="pass",
        frame_idx=10,
        timestamp=0.33,
        confidence=0.72,
        metadata={"from_track_id": 3, "to_track_id": 8},
    )

    assert event.metadata is not None
    assert event.metadata["schema_version"] == EVENT_METADATA_SCHEMA_VERSION
    assert event.metadata["event_family"] == "pass"
    assert event.metadata["event_type"] == "pass"


def test_set_piece_event_metadata_is_schema_versioned() -> None:
    """Set-piece subtypes should include canonical set-piece metadata."""
    event = Event(
        event_type="corner_kick",
        frame_idx=25,
        timestamp=0.83,
        confidence=0.68,
        metadata={"team_id": "ours"},
    )

    assert event.metadata is not None
    assert event.metadata["schema_version"] == EVENT_METADATA_SCHEMA_VERSION
    assert event.metadata["event_family"] == "set_piece"
    assert event.metadata["event_type"] == "corner_kick"
    assert event.metadata["set_piece_type"] == "corner_kick"
