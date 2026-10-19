import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from schema import Item
from db import ItemDB
from renderer import _normalize_summary, render_today_brief


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated SQLite ItemDB in a temporary directory."""
    db_file = tmp_path / "test_brief.db"
    with ItemDB(db_path=str(db_file)) as db:
        yield db


def test_item_db_context_manager(tmp_path):
    """Verifies that ItemDB works cleanly as a context manager and closes properly."""
    db_file = tmp_path / "ctx_test.db"
    with ItemDB(db_path=str(db_file)) as db:
        assert db.conn is not None
    # After exit, connection should be closed
    with pytest.raises(Exception):
        db.conn.cursor()


def test_item_db_upsert_and_deduplication(temp_db):
    """Tests inserting items and verifying that duplicate IDs are ignored."""
    now = datetime.now(timezone.utc)
    item1 = Item(
        id="arxiv:2401.0001",
        source="arxiv",
        title="Reasoning with LLMs",
        url="https://arxiv.org/abs/2401.0001",
        published_at=now,
        summary="A paper on reasoning.",
        authors=["Alice", "Bob"],
        tags=["cs.AI"]
    )
    new_count, exist_count = temp_db.upsert_items([item1])
    assert new_count == 1
    assert exist_count == 0

    # Insert duplicate
    new_count, exist_count = temp_db.upsert_items([item1])
    assert new_count == 0
    assert exist_count == 1


def test_item_filtering_and_brief_pipeline(temp_db, tmp_path):
    """Tests the full lifecycle: upsert -> filter -> summarize -> render markdown."""
    now = datetime.now(timezone.utc)
    item = Item(
        id="arxiv:2401.0002",
        source="arxiv",
        title="Agentic Workflows in Production",
        url="https://arxiv.org/abs/2401.0002",
        published_at=now,
        summary="Building robust agent loops.",
        authors=["Charlie"],
        tags=["cs.AI"]
    )
    temp_db.upsert_items([item])

    # 1. Update filter pass
    temp_db.update_filter_result(
        item_id=item.id,
        passed=True,
        reason="Highly relevant to autonomous agents",
        category="Agent",
        importance="P0"
    )

    # 2. Update summary
    summary_text = "【核心贡献】提出了新的评估范式。【方法创新】引入轻量沙箱。"
    temp_db.update_summary(item_id=item.id, summary=summary_text)

    # 3. Retrieve today's brief items
    brief_items = temp_db.get_today_brief_items(days_window=5)
    assert len(brief_items) == 1
    assert brief_items[0].id == item.id
    assert brief_items[0].category == "Agent"

    # 4. Render brief
    out_dir = tmp_path / "briefs"
    brief_path = render_today_brief(db=temp_db, output_dir=out_dir, days_window=5)
    assert brief_path is not None
    assert brief_path.exists()
    content = brief_path.read_text(encoding="utf-8")
    assert "Agentic Workflows in Production" in content
    assert "## Agent" in content

    # 5. After rendering, items are marked in brief
    subsequent_items = temp_db.get_today_brief_items(days_window=5)
    assert len(subsequent_items) == 0


def test_normalize_summary_segmentation():
    """Tests that Chinese bracketed segments are properly separated by <br>."""
    raw = "【背景】现有模型长文本受限。【方法】提出流式注意力。【结果】降低 40% 显存。"
    res = _normalize_summary(raw)
    assert "<br>" in res
    assert res.startswith("【背景】")
