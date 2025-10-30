"""
Tests for STATE-SNAPSHOT System (Component 10)
"""

import pytest
import json
from src.v2.state_snapshot import (
    StateSnapshotSystem,
    SystemSnapshot,
    ComponentState,
    SnapshotDiff
)


# Mock component class for testing
class MockComponent:
    def __init__(self, name):
        self.name = name
        self.stats = {
            "total": 0,
            "count": 0
        }
        self.config_value = 100
        self.enabled = True

    def get_stats(self):
        return self.stats.copy()


def test_initialization():
    """Test StateSnapshotSystem initialization"""
    system = StateSnapshotSystem()

    assert system.current_version == "1.0.0"
    assert len(system.snapshots) == 0
    assert system.max_snapshots == 10
    assert system.stats["total_snapshots"] == 0


def test_capture_snapshot():
    """Test capturing system snapshot"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 5
    comp1.stats["count"] = 3

    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)

    assert isinstance(snapshot, SystemSnapshot)
    assert snapshot.version == "1.0.0"
    assert "comp1" in snapshot.components
    assert snapshot.components["comp1"].stats["total"] == 5
    assert system.stats["total_snapshots"] == 1


def test_capture_multiple_components():
    """Test capturing multiple components"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 10

    comp2 = MockComponent("comp2")
    comp2.stats["total"] = 20

    components = {"comp1": comp1, "comp2": comp2}

    snapshot = system.capture_snapshot(components)

    assert len(snapshot.components) == 2
    assert snapshot.components["comp1"].stats["total"] == 10
    assert snapshot.components["comp2"].stats["total"] == 20


def test_capture_with_metadata():
    """Test capturing snapshot with metadata"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    metadata = {
        "user": "test_user",
        "reason": "checkpoint"
    }

    snapshot = system.capture_snapshot(components, metadata=metadata)

    assert snapshot.metadata["user"] == "test_user"
    assert snapshot.metadata["reason"] == "checkpoint"


def test_serialize_json():
    """Test JSON serialization"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)
    json_str = system.serialize_json(snapshot)

    assert isinstance(json_str, str)
    assert "version" in json_str
    assert "comp1" in json_str
    # Should be valid JSON
    data = json.loads(json_str)
    assert data["version"] == "1.0.0"


def test_deserialize_json():
    """Test JSON deserialization"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 42
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)
    json_str = system.serialize_json(snapshot)

    # Deserialize
    restored = system.deserialize_json(json_str)

    assert isinstance(restored, SystemSnapshot)
    assert restored.version == snapshot.version
    assert "comp1" in restored.components
    assert restored.components["comp1"].stats["total"] == 42


def test_serialize_pickle():
    """Test pickle serialization"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)
    pickled = system.serialize_pickle(snapshot)

    assert isinstance(pickled, bytes)
    assert len(pickled) > 0


def test_deserialize_pickle():
    """Test pickle deserialization"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 99
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)
    pickled = system.serialize_pickle(snapshot)

    # Deserialize
    restored = system.deserialize_pickle(pickled)

    assert isinstance(restored, SystemSnapshot)
    assert restored.version == snapshot.version
    assert restored.components["comp1"].stats["total"] == 99


def test_compress_snapshot():
    """Test snapshot compression"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)

    # Get original size
    original = system.serialize_json(snapshot).encode('utf-8')

    # Compress
    compressed = system.compress_snapshot(snapshot)

    assert isinstance(compressed, bytes)
    # Compressed should be smaller (or at least different)
    assert len(compressed) <= len(original)


def test_decompress_snapshot():
    """Test snapshot decompression"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 123
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)
    compressed = system.compress_snapshot(snapshot)

    # Decompress
    restored = system.decompress_snapshot(compressed)

    assert isinstance(restored, SystemSnapshot)
    assert restored.version == snapshot.version
    assert restored.components["comp1"].stats["total"] == 123


def test_calculate_diff_no_changes():
    """Test diff calculation with no changes"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    snapshot1 = system.capture_snapshot(components)
    snapshot2 = system.capture_snapshot(components)

    diff = system.calculate_diff(snapshot1, snapshot2)

    assert isinstance(diff, SnapshotDiff)
    assert len(diff.added_components) == 0
    assert len(diff.removed_components) == 0
    assert len(diff.modified_components) == 0


def test_calculate_diff_with_changes():
    """Test diff calculation with modifications"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 10
    components = {"comp1": comp1}

    snapshot1 = system.capture_snapshot(components)

    # Modify component
    comp1.stats["total"] = 20
    comp1.stats["count"] = 5

    snapshot2 = system.capture_snapshot(components)

    diff = system.calculate_diff(snapshot1, snapshot2)

    assert len(diff.modified_components) == 1
    assert "comp1" in diff.modified_components
    assert diff.modified_components["comp1"]["stats"]["total"]["old"] == 10
    assert diff.modified_components["comp1"]["stats"]["total"]["new"] == 20


def test_calculate_diff_added_component():
    """Test diff with added component"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components1 = {"comp1": comp1}

    snapshot1 = system.capture_snapshot(components1)

    # Add new component
    comp2 = MockComponent("comp2")
    components2 = {"comp1": comp1, "comp2": comp2}

    snapshot2 = system.capture_snapshot(components2)

    diff = system.calculate_diff(snapshot1, snapshot2)

    assert "comp2" in diff.added_components
    assert len(diff.removed_components) == 0


def test_calculate_diff_removed_component():
    """Test diff with removed component"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp2 = MockComponent("comp2")
    components1 = {"comp1": comp1, "comp2": comp2}

    snapshot1 = system.capture_snapshot(components1)

    # Remove component
    components2 = {"comp1": comp1}

    snapshot2 = system.capture_snapshot(components2)

    diff = system.calculate_diff(snapshot1, snapshot2)

    assert "comp2" in diff.removed_components
    assert len(diff.added_components) == 0


def test_restore_snapshot():
    """Test restoring snapshot"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 50
    comp1.config_value = 200
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)

    # Modify component
    comp1.stats["total"] = 100
    comp1.config_value = 300

    # Restore
    success = system.restore_snapshot(snapshot, components)

    assert success is True
    assert comp1.stats["total"] == 50
    assert comp1.config_value == 200
    assert system.stats["total_restorations"] == 1


def test_rollback():
    """Test rollback functionality"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    # Create first snapshot
    comp1.stats["total"] = 10
    snapshot1 = system.capture_snapshot(components)

    # Create second snapshot
    comp1.stats["total"] = 20
    snapshot2 = system.capture_snapshot(components)

    # Create third snapshot
    comp1.stats["total"] = 30
    snapshot3 = system.capture_snapshot(components)

    # Rollback 1 step
    restored = system.rollback(components, steps=1)

    assert restored is not None
    assert comp1.stats["total"] == 20
    assert system.stats["total_rollbacks"] == 1


def test_rollback_multiple_steps():
    """Test rollback multiple steps"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    # Create snapshots
    for i in range(5):
        comp1.stats["total"] = i * 10
        system.capture_snapshot(components)

    # Rollback 3 steps
    restored = system.rollback(components, steps=3)

    assert restored is not None
    assert comp1.stats["total"] == 10  # Should be at step 1


def test_rollback_too_far():
    """Test rollback beyond available history"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    # Only create 2 snapshots
    system.capture_snapshot(components)
    system.capture_snapshot(components)

    # Try to rollback 5 steps (too far)
    restored = system.rollback(components, steps=5)

    assert restored is None


def test_get_latest_snapshot():
    """Test getting latest snapshot"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    # Create snapshots
    comp1.stats["total"] = 10
    snapshot1 = system.capture_snapshot(components)

    comp1.stats["total"] = 20
    snapshot2 = system.capture_snapshot(components)

    latest = system.get_latest_snapshot()

    assert latest is not None
    assert latest.components["comp1"].stats["total"] == 20


def test_list_snapshots():
    """Test listing snapshots"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    # Create multiple snapshots
    for i in range(3):
        system.capture_snapshot(components)

    snapshots = system.list_snapshots()

    assert len(snapshots) == 3
    assert all("version" in s for s in snapshots)
    assert all("timestamp" in s for s in snapshots)
    assert all("component_count" in s for s in snapshots)


def test_max_snapshot_limit():
    """Test max snapshot limit enforcement"""
    system = StateSnapshotSystem()
    system.max_snapshots = 5

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    # Create more snapshots than limit
    for i in range(10):
        comp1.stats["total"] = i
        system.capture_snapshot(components)

    # Should only keep last 5
    assert len(system.snapshots) == 5
    # Should have the latest ones (5-9)
    assert system.snapshots[-1].components["comp1"].stats["total"] == 9


def test_clear_history():
    """Test clearing snapshot history"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    # Create snapshots
    for i in range(3):
        system.capture_snapshot(components)

    assert len(system.snapshots) == 3

    # Clear
    system.clear_history()

    assert len(system.snapshots) == 0


def test_statistics_tracking():
    """Test statistics tracking"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    # Perform operations
    snapshot1 = system.capture_snapshot(components)
    snapshot2 = system.capture_snapshot(components)

    system.restore_snapshot(snapshot1, components)

    diff = system.calculate_diff(snapshot1, snapshot2)

    system.rollback(components, steps=1)

    stats = system.get_stats()

    assert stats["total_snapshots"] == 2
    # Restorations: 1 direct + 1 from rollback = 2
    assert stats["total_restorations"] == 2
    assert stats["total_diffs"] == 1
    assert stats["total_rollbacks"] == 1
    assert stats["current_snapshots"] == 2


def test_stats_reset():
    """Test statistics reset"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    system.capture_snapshot(components)
    system.capture_snapshot(components)

    assert system.stats["total_snapshots"] == 2

    system.reset_stats()

    assert system.stats["total_snapshots"] == 0


def test_export_import_json():
    """Test export and import in JSON format"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 777
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)

    # Export
    exported = system.export_snapshot(snapshot, format="json", compress=False)

    assert isinstance(exported, bytes)

    # Import
    imported = system.import_snapshot(exported, format="json", compressed=False)

    assert isinstance(imported, SystemSnapshot)
    assert imported.components["comp1"].stats["total"] == 777


def test_export_import_pickle():
    """Test export and import in pickle format"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 888
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)

    # Export
    exported = system.export_snapshot(snapshot, format="pickle", compress=False)

    assert isinstance(exported, bytes)

    # Import
    imported = system.import_snapshot(exported, format="pickle", compressed=False)

    assert isinstance(imported, SystemSnapshot)
    assert imported.components["comp1"].stats["total"] == 888


def test_export_import_compressed():
    """Test export and import with compression"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 999
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)

    # Export compressed
    exported = system.export_snapshot(snapshot, format="json", compress=True)

    # Import compressed
    imported = system.import_snapshot(exported, format="json", compressed=True)

    assert isinstance(imported, SystemSnapshot)
    assert imported.components["comp1"].stats["total"] == 999


def test_snapshot_versioning():
    """Test snapshot versioning"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)

    assert snapshot.version == "1.0.0"

    # Change version
    system.current_version = "2.0.0"

    snapshot2 = system.capture_snapshot(components)

    assert snapshot2.version == "2.0.0"


def test_get_snapshot_by_version():
    """Test retrieving snapshot by version"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    # Create snapshot with current version
    snapshot = system.capture_snapshot(components)

    # Retrieve by version
    found = system.get_snapshot_by_version("1.0.0")

    assert found is not None
    assert found.version == "1.0.0"


def test_get_snapshot_by_nonexistent_version():
    """Test retrieving nonexistent snapshot version"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    system.capture_snapshot(components)

    # Try to get nonexistent version
    found = system.get_snapshot_by_version("99.99.99")

    assert found is None


def test_component_state_extraction():
    """Test component state extraction"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    comp1.stats["total"] = 100
    comp1.config_value = 500
    comp1.enabled = False

    components = {"comp1": comp1}

    snapshot = system.capture_snapshot(components)

    state = snapshot.components["comp1"]

    assert state.stats["total"] == 100
    assert state.config["config_value"] == 500
    assert state.config["enabled"] is False


def test_multiple_restoration_cycles():
    """Test multiple save/restore cycles"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    # Cycle 1
    comp1.stats["total"] = 10
    snapshot1 = system.capture_snapshot(components)

    comp1.stats["total"] = 20
    system.restore_snapshot(snapshot1, components)
    assert comp1.stats["total"] == 10

    # Cycle 2
    comp1.stats["total"] = 30
    snapshot2 = system.capture_snapshot(components)

    comp1.stats["total"] = 40
    system.restore_snapshot(snapshot2, components)
    assert comp1.stats["total"] == 30


def test_empty_components():
    """Test snapshot with no components"""
    system = StateSnapshotSystem()

    components = {}

    snapshot = system.capture_snapshot(components)

    assert len(snapshot.components) == 0
    assert isinstance(snapshot, SystemSnapshot)


def test_diff_stats_tracking():
    """Test diff statistics tracking"""
    system = StateSnapshotSystem()

    comp1 = MockComponent("comp1")
    components = {"comp1": comp1}

    snapshot1 = system.capture_snapshot(components)
    snapshot2 = system.capture_snapshot(components)

    assert system.stats["total_diffs"] == 0

    system.calculate_diff(snapshot1, snapshot2)

    assert system.stats["total_diffs"] == 1
