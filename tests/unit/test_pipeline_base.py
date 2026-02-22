"""Tests for pipeline orchestration and stage management."""

import json
import pytest
from pathlib import Path

from src.pipeline.base import (
    Pipeline,
    PipelineCancelledError,
    PipelineStage,
    save_detections_to_parquet,
    save_detections_to_jsonl,
)
from src.config.schemas import PipelineConfig


# -----------------------------------------------------------------------------
# Mock Stage for Testing
# -----------------------------------------------------------------------------

class MockStage(PipelineStage):
    """Mock pipeline stage for testing."""

    def __init__(self, name: str, config: PipelineConfig, result: dict = None):
        super().__init__(name, config)
        self.result = result or {}
        self.run_called = False
        self.context_received = None

    def run(self, context: dict) -> dict:
        self.run_called = True
        self.context_received = context.copy()
        context.update(self.result)
        return context


# -----------------------------------------------------------------------------
# PipelineStage Tests
# -----------------------------------------------------------------------------

class TestPipelineStage:
    """Tests for PipelineStage base class."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a pipeline config."""
        return PipelineConfig(
            cache_dir=str(tmp_path / "cache"),
            enable_cache=True,
        )

    @pytest.fixture
    def stage(self, config):
        """Create a mock stage."""
        return MockStage("test_stage", config, result={"test_key": "test_value"})

    def test_stage_initialization(self, stage, config):
        """Test stage is initialized correctly."""
        assert stage.name == "test_stage"
        assert stage.config == config

    def test_get_cache_key_deterministic(self, stage):
        """Test cache key generation is deterministic."""
        inputs = {"a": 1, "b": "test"}

        key1 = stage.get_cache_key(inputs)
        key2 = stage.get_cache_key(inputs)

        assert key1 == key2

    def test_get_cache_key_different_inputs(self, stage):
        """Test different inputs produce different cache keys."""
        inputs1 = {"a": 1}
        inputs2 = {"a": 2}

        key1 = stage.get_cache_key(inputs1)
        key2 = stage.get_cache_key(inputs2)

        assert key1 != key2

    def test_get_cache_key_order_independent(self, stage):
        """Test cache key is independent of key order."""
        inputs1 = {"a": 1, "b": 2}
        inputs2 = {"b": 2, "a": 1}

        key1 = stage.get_cache_key(inputs1)
        key2 = stage.get_cache_key(inputs2)

        assert key1 == key2

    def test_get_cache_path(self, stage, tmp_path):
        """Test cache path generation."""
        cache_key = "abc123"

        path = stage.get_cache_path(cache_key)

        assert path.parent.name == "test_stage"
        assert path.name == "abc123.json"

    def test_get_cache_path_custom_suffix(self, stage):
        """Test cache path with custom suffix."""
        cache_key = "abc123"

        path = stage.get_cache_path(cache_key, suffix=".parquet")

        assert path.name == "abc123.parquet"

    def test_save_and_load_cache(self, stage):
        """Test saving and loading from cache."""
        cache_key = "test_key_123"
        result = {"foo": "bar", "count": 42}

        stage.save_to_cache(cache_key, result)
        loaded = stage.load_from_cache(cache_key)

        assert loaded == result

    def test_load_from_cache_nonexistent(self, stage):
        """Test loading nonexistent cache returns None."""
        loaded = stage.load_from_cache("nonexistent_key")

        assert loaded is None

    def test_cache_disabled(self, tmp_path):
        """Test caching is disabled when config says so."""
        config = PipelineConfig(
            cache_dir=str(tmp_path / "cache"),
            enable_cache=False,
        )
        stage = MockStage("test", config)

        cache_key = "test_key"
        result = {"data": "test"}

        # Save should not write
        stage.save_to_cache(cache_key, result)

        # Load should return None (even if file existed)
        loaded = stage.load_from_cache(cache_key)

        assert loaded is None

    def test_check_cancelled_noop_without_callback(self, stage):
        """check_cancelled is a no-op when no callback is in context."""
        context = {"video_path": "test.mp4"}
        # Should not raise
        stage.check_cancelled(context)

    def test_check_cancelled_no_raise_when_false(self, stage):
        """check_cancelled does not raise when callback returns False."""
        context = {"_check_cancel": lambda: False}
        stage.check_cancelled(context)

    def test_check_cancelled_raises_when_true(self, stage):
        """check_cancelled raises PipelineCancelledError when callback returns True."""
        context = {"_check_cancel": lambda: True}
        with pytest.raises(PipelineCancelledError):
            stage.check_cancelled(context)


# -----------------------------------------------------------------------------
# Pipeline Tests
# -----------------------------------------------------------------------------

class TestPipeline:
    """Tests for Pipeline class."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a pipeline config."""
        return PipelineConfig(
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "output"),
        )

    @pytest.fixture
    def pipeline(self, config):
        """Create a pipeline."""
        return Pipeline(config)

    def test_pipeline_initialization(self, pipeline, config):
        """Test pipeline is initialized correctly."""
        assert pipeline.config == config
        assert pipeline.stages == []

    def test_add_stage(self, pipeline, config):
        """Test adding stages to pipeline."""
        stage1 = MockStage("stage1", config)
        stage2 = MockStage("stage2", config)

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)

        assert len(pipeline.stages) == 2
        assert pipeline.stages[0] is stage1
        assert pipeline.stages[1] is stage2

    def test_run_empty_pipeline(self, pipeline, tmp_path):
        """Test running pipeline with no stages."""
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        context = pipeline.run(video_path, output_dir)

        assert "video_path" in context
        assert "output_dir" in context
        assert "start_time" in context
        assert "end_time" in context

    def test_run_executes_stages(self, pipeline, config, tmp_path):
        """Test that run executes all stages."""
        stage1 = MockStage("stage1", config, result={"s1": "done"})
        stage2 = MockStage("stage2", config, result={"s2": "done"})

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        context = pipeline.run(video_path, output_dir)

        assert stage1.run_called
        assert stage2.run_called
        assert context["s1"] == "done"
        assert context["s2"] == "done"

    def test_run_passes_context_between_stages(self, pipeline, config, tmp_path):
        """Test that context is passed between stages."""
        stage1 = MockStage("stage1", config, result={"from_stage1": "data"})
        stage2 = MockStage("stage2", config)

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        pipeline.run(video_path, output_dir)

        # Stage 2 should have received stage 1's output
        assert "from_stage1" in stage2.context_received

    def test_run_creates_output_dir(self, pipeline, tmp_path):
        """Test that run creates output directory."""
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "new_output_dir"

        pipeline.run(video_path, output_dir)

        assert output_dir.exists()

    def test_run_saves_manifest(self, pipeline, config, tmp_path):
        """Test that run saves manifest."""
        stage = MockStage("test", config)
        pipeline.add_stage(stage)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        pipeline.run(video_path, output_dir)

        manifest_path = output_dir / "run_manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "schema_version" in manifest
        assert "video_path" in manifest
        assert "stages" in manifest
        assert "test" in manifest["stages"]

    def test_run_saves_summary_and_ui_index(self, pipeline, config, tmp_path):
        """Test that run saves summary.json and ui_index.json artifacts."""

        class ArtifactStage(PipelineStage):
            def run(self, context):
                output_dir = Path(context["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)

                with open(output_dir / "events.jsonl", "w") as f:
                    f.write(
                        json.dumps(
                            {
                                "event_type": "goal",
                                "frame_idx": 30,
                                "timestamp": 1.0,
                                "confidence": 0.9,
                                "location": None,
                                "metadata": {},
                            }
                        )
                        + "\n"
                    )

                with open(output_dir / "score_timeline.json", "w") as f:
                    json.dump(
                        {
                            "goals": 1,
                            "final_score": {"team_a": 1, "team_b": 0},
                            "timeline": [
                                {
                                    "timestamp": 1.0,
                                    "frame_idx": 30,
                                    "score": {"team_a": 1, "team_b": 0},
                                    "confidence": 0.9,
                                    "goal_region": "top",
                                }
                            ],
                        },
                        f,
                        indent=2,
                    )

                with open(output_dir / "highlights.json", "w") as f:
                    json.dump({"schema_version": "1.0", "segments": []}, f, indent=2)

                with open(output_dir / "player_analytics.json", "w") as f:
                    json.dump(
                        {
                            "schema_version": "1.0",
                            "summary": {
                                "runs_analyzed": 2,
                                "players_detected": 3,
                                "events_total": 6,
                                "sprints_total": 4,
                            },
                            "runs": [],
                            "players": [],
                        },
                        f,
                        indent=2,
                    )

                with open(output_dir / "player_highlights.json", "w") as f:
                    json.dump(
                        {
                            "schema_version": "1.0",
                            "players": [],
                            "summary": {
                                "players_with_reels": 0,
                                "player_segments_total": 0,
                            },
                        },
                        f,
                        indent=2,
                    )
                with open(output_dir / "cross_match_report.json", "w") as f:
                    json.dump(
                        {
                            "schema_version": "1.0",
                            "summary": {
                                "matches_analyzed": 3,
                                "unique_players": 8,
                            },
                        },
                        f,
                        indent=2,
                    )

                context["video_metadata"] = {
                    "fps": 30.0,
                    "duration": 10.0,
                    "total_frames": 300,
                    "width": 1280,
                    "height": 720,
                }
                context["events"] = [{"event_type": "goal"}]
                context["score_timeline"] = [{"timestamp": 1.0}]
                context["detection_custom_metrics"] = {
                    "total_detections": 25,
                    "player_detections": 20,
                    "ball_detections": 5,
                }
                context["tracking_custom_metrics"] = {
                    "track_points": 40,
                    "unique_tracks": 7,
                }
                context["event_detection_custom_metrics"] = {
                    "shots": 2,
                    "goals": 1,
                }
                context["highlight_generation_custom_metrics"] = {
                    "segments_selected": 3,
                    "candidates": 6,
                }
                context["player_analytics_custom_metrics"] = {
                    "runs_analyzed": 2,
                    "players_detected": 3,
                    "events_total": 6,
                    "sprints_total": 4,
                }
                context["player_highlight_reels_custom_metrics"] = {
                    "players_with_reels": 1,
                    "player_segments_total": 3,
                }
                context["cross_match_reporting_custom_metrics"] = {
                    "matches_analyzed": 3,
                    "unique_players": 8,
                }
                return context

        pipeline.add_stage(ArtifactStage("artifact_stage", config))

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        pipeline.run(video_path, output_dir)

        summary_path = output_dir / "summary.json"
        ui_index_path = output_dir / "ui_index.json"
        assert summary_path.exists()
        assert ui_index_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)
        with open(ui_index_path) as f:
            ui_index = json.load(f)

        assert summary["schema_version"] == "1.0"
        assert summary["run_name"] == "output"
        assert summary["counts"]["detections_total"] == 25
        assert summary["counts"]["goals"] == 1
        assert summary["counts"]["cross_match_matches"] == 3
        assert summary["score"]["final_score"] == {"team_a": 1, "team_b": 0}
        assert summary["artifacts"]["run_manifest"] == "run_manifest.json"
        assert summary["artifacts"]["summary"] == "summary.json"
        assert summary["artifacts"]["player_analytics"] == "player_analytics.json"
        assert summary["artifacts"]["cross_match_report"] == "cross_match_report.json"

        assert ui_index["schema_version"] == "1.0"
        assert ui_index["run_name"] == "output"
        assert ui_index["summary_path"] == "summary.json"
        assert ui_index["artifacts"]["ui_index"] == "ui_index.json"
        assert ui_index["capabilities"]["has_events"] is True
        assert ui_index["capabilities"]["has_player_analytics"] is True
        assert ui_index["capabilities"]["has_highlights"] is True
        assert ui_index["capabilities"]["has_player_reels"] is True
        assert ui_index["capabilities"]["has_cross_match_report"] is True

    def test_summary_and_ui_index_include_match_stats_and_coach_assist_artifacts(self, pipeline, config, tmp_path):
        """Summary/UI index should expose match_stats and coach_assist artifacts when present."""

        class MatchStatsStage(PipelineStage):
            def run(self, context):
                output_dir = Path(context["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)

                with open(output_dir / "match_stats.json", "w") as f:
                    json.dump(
                        {
                            "schema_version": "1.0",
                            "teams": {"ours": {"shots": 2, "goals": 1}},
                            "totals": {"shots": 2, "goals": 1, "passes": 0, "set_pieces": 0},
                        },
                        f,
                        indent=2,
                    )
                with open(output_dir / "coach_assist.json", "w") as f:
                    json.dump(
                        {
                            "schema_version": "1.0",
                            "enabled": True,
                            "provider": "heuristic",
                            "summary": {"status": "ready", "insights_generated": 1},
                            "insights": [{"insight_id": "insight_001", "title": "Example"}],
                        },
                        f,
                        indent=2,
                    )

                context["video_metadata"] = {
                    "fps": 30.0,
                    "duration": 10.0,
                    "total_frames": 300,
                    "width": 1280,
                    "height": 720,
                }
                context["events"] = []
                return context

        pipeline.add_stage(MatchStatsStage("match_stats_stage", config))

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        pipeline.run(video_path, output_dir)

        with open(output_dir / "summary.json") as f:
            summary = json.load(f)
        with open(output_dir / "ui_index.json") as f:
            ui_index = json.load(f)

        assert summary["artifacts"]["match_stats"] == "match_stats.json"
        assert summary["artifacts"]["coach_assist"] == "coach_assist.json"
        assert ui_index["capabilities"]["has_match_stats"] is True
        assert ui_index["capabilities"]["has_coach_assist"] is True

    def test_run_resume_mode(self, pipeline, config, tmp_path):
        """Test resume mode is passed in context when sentinel exists."""
        stage = MockStage("test", config)
        pipeline.add_stage(stage)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "video_metadata.json").write_text("{}")

        pipeline.run(video_path, output_dir, resume=True)

        assert stage.context_received["resume"] is True

    def test_resume_downgraded_when_no_artifacts(self, pipeline, config, tmp_path):
        """resume=True on empty dir → stage sees context['resume'] == False."""
        stage = MockStage("test", config)
        pipeline.add_stage(stage)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"
        # Do NOT create video_metadata.json sentinel

        pipeline.run(video_path, output_dir, resume=True)

        assert stage.context_received["resume"] is False

    def test_resume_preserved_when_sentinel_exists(self, pipeline, config, tmp_path):
        """Create video_metadata.json first → stage sees context['resume'] == True."""
        stage = MockStage("test", config)
        pipeline.add_stage(stage)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "video_metadata.json").write_text("{}")

        pipeline.run(video_path, output_dir, resume=True)

        assert stage.context_received["resume"] is True

    def test_output_dir_recreated_between_stages(self, pipeline, config, tmp_path):
        """First stage deletes output_dir, second stage still runs without crash."""
        import shutil

        class DirDeletingStage(PipelineStage):
            def run(self, context):
                shutil.rmtree(context["output_dir"])
                return context

        stage1 = DirDeletingStage("deleter", config)
        stage2 = MockStage("survivor", config, result={"survived": True})

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        context = pipeline.run(video_path, output_dir)

        assert stage2.run_called
        assert context["survived"] is True
        assert output_dir.exists()

    def test_run_stage_error_propagates(self, pipeline, config, tmp_path):
        """Test that stage errors are propagated."""
        class ErrorStage(PipelineStage):
            def run(self, context):
                raise ValueError("Stage failed!")

        pipeline.add_stage(ErrorStage("error_stage", config))

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="Stage failed!"):
            pipeline.run(video_path, output_dir)


class TestPipelineManifest:
    """Tests for manifest saving."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a pipeline config."""
        return PipelineConfig(cache_dir=str(tmp_path / "cache"))

    def test_manifest_includes_config(self, config, tmp_path):
        """Test manifest includes config dump."""
        pipeline = Pipeline(config)
        stage = MockStage("test", config)
        pipeline.add_stage(stage)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        pipeline.run(video_path, output_dir)

        manifest_path = output_dir / "run_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "config" in manifest

    def test_manifest_includes_times(self, config, tmp_path):
        """Test manifest includes start and end times."""
        pipeline = Pipeline(config)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"

        pipeline.run(video_path, output_dir)

        manifest_path = output_dir / "run_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "start_time" in manifest
        assert "end_time" in manifest


# -----------------------------------------------------------------------------
# Detection Export Tests
# -----------------------------------------------------------------------------

class TestSaveDetections:
    """Tests for detection export functions."""

    @pytest.fixture
    def sample_detections(self):
        """Create sample detections."""
        return [
            {
                "frame_idx": 0,
                "track_id": 1,
                "bbox_x1": 100.0,
                "bbox_y1": 100.0,
                "bbox_x2": 150.0,
                "bbox_y2": 200.0,
                "confidence": 0.9,
                "object_type": "player",
            },
            {
                "frame_idx": 0,
                "track_id": 2,
                "bbox_x1": 200.0,
                "bbox_y1": 100.0,
                "bbox_x2": 250.0,
                "bbox_y2": 200.0,
                "confidence": 0.85,
                "object_type": "player",
            },
            {
                "frame_idx": 1,
                "track_id": 1,
                "bbox_x1": 105.0,
                "bbox_y1": 102.0,
                "bbox_x2": 155.0,
                "bbox_y2": 202.0,
                "confidence": 0.88,
                "object_type": "player",
            },
        ]

    def test_save_to_parquet(self, sample_detections, tmp_path):
        """Test saving detections to parquet."""
        output_path = tmp_path / "detections.parquet"

        save_detections_to_parquet(sample_detections, output_path)

        assert output_path.exists()

        # Verify can be read back
        import pandas as pd
        df = pd.read_parquet(output_path)
        assert len(df) == 3
        assert "frame_idx" in df.columns
        assert "track_id" in df.columns

    def test_save_to_jsonl(self, sample_detections, tmp_path):
        """Test saving detections to JSONL."""
        output_path = tmp_path / "detections.jsonl"

        save_detections_to_jsonl(sample_detections, output_path)

        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "frame_idx" in data

    def test_save_empty_detections_parquet(self, tmp_path):
        """Test saving empty detections to parquet."""
        output_path = tmp_path / "empty.parquet"

        save_detections_to_parquet([], output_path)

        assert output_path.exists()

    def test_save_empty_detections_jsonl(self, tmp_path):
        """Test saving empty detections to JSONL."""
        output_path = tmp_path / "empty.jsonl"

        save_detections_to_jsonl([], output_path)

        assert output_path.exists()

        with open(output_path) as f:
            content = f.read()
        assert content == ""

    def test_parquet_preserves_types(self, tmp_path):
        """Test that parquet preserves data types."""
        detections = [
            {
                "frame_idx": 100,  # int
                "confidence": 0.95,  # float
                "object_type": "player",  # str
            }
        ]

        output_path = tmp_path / "typed.parquet"
        save_detections_to_parquet(detections, output_path)

        import pandas as pd
        df = pd.read_parquet(output_path)

        assert df["frame_idx"].iloc[0] == 100
        assert df["confidence"].iloc[0] == pytest.approx(0.95)
        assert df["object_type"].iloc[0] == "player"

    def test_jsonl_preserves_values(self, tmp_path):
        """Test that JSONL preserves all values."""
        detections = [
            {
                "frame_idx": 100,
                "confidence": 0.95,
                "metadata": {"key": "value"},
            }
        ]

        output_path = tmp_path / "test.jsonl"
        save_detections_to_jsonl(detections, output_path)

        with open(output_path) as f:
            loaded = json.loads(f.readline())

        assert loaded["frame_idx"] == 100
        assert loaded["confidence"] == pytest.approx(0.95)
        assert loaded["metadata"]["key"] == "value"
