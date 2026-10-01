from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile

import pytest

from agent_workflow.core.entities.models.skills import (
    Skill,
    SkillMetadata,
    SkillRegistry,
    SkillLoader,
    SkillManager,
    SkillEvent,
    SkillEventType,
    SkillRegistryError,
    SkillNotFoundError,
    SkillAlreadyRegisteredError,
    InvalidSkillError,
    SkillLoadError,
    parse_skill_file,
)


class TestSkillModel:
    def test_valid_skill(self) -> None:
        skill = Skill(name="test", description="Test skill", instructions="Do something")
        assert skill.name == "test"
        assert skill.description == "Test skill"
        assert skill.version == "1.0.0"
        assert skill.instructions == "Do something"

    def test_skill_with_metadata(self) -> None:
        skill = Skill(
            name="test",
            description="Test skill",
            version="2.0.0",
            metadata={"tags": ["test"], "author": "me"},
        )
        assert skill.version == "2.0.0"
        assert skill.metadata["tags"] == ["test"]
        assert skill.metadata["author"] == "me"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Skill name cannot be empty"):
            Skill(name="", description="Test")

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="Skill description cannot be empty"):
            Skill(name="test", description="")

    def test_summary(self) -> None:
        skill = Skill(name="test", description="First line\nSecond line")
        assert skill.summary() == "First line"


class TestSkillRegistry:
    def test_register_and_get(self) -> None:
        registry = SkillRegistry()
        skill = Skill(name="test", description="Test skill")
        registry.register(skill)

        assert registry.has("test")
        assert registry.get("test") == skill

    def test_list(self) -> None:
        registry = SkillRegistry()
        skill1 = Skill(name="a", description="A")
        skill2 = Skill(name="b", description="B")
        registry.register(skill1)
        registry.register(skill2)

        skills = registry.list()
        assert len(skills) == 2
        assert {s.name for s in skills} == {"a", "b"}

    def test_list_metadata(self) -> None:
        registry = SkillRegistry()
        skill = Skill(name="test", description="Test skill", version="1.5.0", metadata={"tag": "test"})
        registry.register(skill)

        metadata = registry.list_metadata()
        assert len(metadata) == 1
        assert metadata[0].name == "test"
        assert metadata[0].version == "1.5.0"
        assert metadata[0].metadata["tag"] == "test"

    def test_unregister(self) -> None:
        registry = SkillRegistry()
        skill = Skill(name="test", description="Test")
        registry.register(skill)

        registry.unregister("test")
        assert not registry.has("test")

    def test_duplicate_registration_raises(self) -> None:
        registry = SkillRegistry()
        skill = Skill(name="test", description="Test")
        registry.register(skill)

        with pytest.raises(SkillAlreadyRegisteredError):
            registry.register(skill)

    def test_get_not_found_raises(self) -> None:
        registry = SkillRegistry()
        with pytest.raises(SkillNotFoundError):
            registry.get("nonexistent")

    def test_unregister_not_found_raises(self) -> None:
        registry = SkillRegistry()
        with pytest.raises(SkillNotFoundError):
            registry.unregister("nonexistent")

    def test_find(self) -> None:
        registry = SkillRegistry()
        skill1 = Skill(name="git-review", description="Review git changes")
        skill2 = Skill(name="python-test", description="Test python code")
        registry.register(skill1)
        registry.register(skill2)

        results = registry.find("git")
        assert len(results) == 1
        assert results[0].name == "git-review"

        results = registry.find("python")
        assert len(results) == 1
        assert results[0].name == "python-test"

        results = registry.find("test")
        assert len(results) == 1
        assert results[0].name == "python-test"

        results = registry.find("review")
        assert len(results) == 1
        assert results[0].name == "git-review"


class TestSkillLoader:
    def test_parse_valid_skill_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test_skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("""---
name: loader-test
description: Test skill from loader
version: 1.0.0
tags: [test]
---

# Test Skill

Instructions for testing.
""")

            skill = parse_skill_file(skill_file)
            assert skill.name == "loader-test"
            assert skill.description == "Test skill from loader"
            assert skill.version == "1.0.0"
            assert skill.metadata["tags"] == ["test"]
            assert "Instructions for testing" in skill.instructions

    def test_missing_frontmatter_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("# No frontmatter\n\nJust content")

            with pytest.raises(SkillLoadError, match="Missing frontmatter"):
                parse_skill_file(skill_file)

    def test_missing_name_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("""---
description: Test skill
---

Content
""")

            with pytest.raises(SkillLoadError, match="Missing required field: name"):
                parse_skill_file(skill_file)

    def test_missing_description_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("""---
name: test
---

Content
""")

            with pytest.raises(SkillLoadError, match="Missing required field: description"):
                parse_skill_file(skill_file)

    def test_empty_name_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("""---
name: ""
description: Test
---

Content
""")

            with pytest.raises(SkillLoadError, match="Required field 'name' cannot be empty"):
                parse_skill_file(skill_file)

    def test_invalid_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("""---
name: test
description: test
invalid: [unclosed
---

Content
""")

            with pytest.raises(SkillLoadError, match="Invalid YAML"):
                parse_skill_file(skill_file)

    def test_load_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            
            skill1_dir = skills_dir / "skill1"
            skill1_dir.mkdir()
            (skill1_dir / "SKILL.md").write_text("""---
name: skill1
description: First skill
---
Content 1
""")

            skill2_dir = skills_dir / "skill2"
            skill2_dir.mkdir()
            (skill2_dir / "SKILL.md").write_text("""---
name: skill2
description: Second skill
---
Content 2
""")

            registry = SkillRegistry()
            loader = SkillLoader(registry)
            skills = loader.load_directory(skills_dir)

            assert len(skills) == 2
            assert {s.name for s in skills} == {"skill1", "skill2"}

    def test_discover(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            
            skill_dir = skills_dir / "test_skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("""---
name: discovered-skill
description: A discovered skill
version: 2.0.0
tags: [test]
---
Content
""")

            registry = SkillRegistry()
            loader = SkillLoader(registry)
            discovered = loader.discover(skills_dir)

            assert len(discovered) == 1
            assert discovered[0].name == "discovered-skill"
            assert discovered[0].version == "2.0.0"
            assert discovered[0].metadata["tags"] == ["test"]


class TestSkillManager:
    @pytest.fixture
    def skills_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            
            # Create test skills
            for name, desc in [
                ("skill-a", "Skill A description"),
                ("skill-b", "Skill B description"),
            ]:
                skill_dir = skills_dir / name
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {desc}
version: 1.0.0
---
# {name}

Instructions for {name}.
""")
            
            yield skills_dir

    @pytest.mark.asyncio
    async def test_discover(self, skills_dir: Path) -> None:
        events = []
        async def on_event(event: SkillEvent):
            events.append(event)

        manager = SkillManager(skills_directory=skills_dir, on_event=on_event)
        skills = await manager.discover()

        assert len(skills) == 2
        assert {s.name for s in skills} == {"skill-a", "skill-b"}
        assert len(events) == 2
        assert all(e.event_type == SkillEventType.DISCOVERED for e in events)

    @pytest.mark.asyncio
    async def test_list_available_metadata(self, skills_dir: Path) -> None:
        manager = SkillManager(skills_directory=skills_dir)
        metadata = await manager.list_available_metadata()

        assert len(metadata) == 2
        names = {m["name"] for m in metadata}
        assert names == {"skill-a", "skill-b"}

    @pytest.mark.asyncio
    async def test_load_skill(self, skills_dir: Path) -> None:
        events = []
        async def on_event(event: SkillEvent):
            events.append(event)

        manager = SkillManager(skills_directory=skills_dir, on_event=on_event)
        skill = await manager.load("skill-a", run_id="run-1")

        assert skill.name == "skill-a"
        assert "skill-a" in manager.loaded_skills
        
        # Check events
        loaded_events = [e for e in events if e.event_type == SkillEventType.LOADED]
        assert len(loaded_events) == 1
        assert loaded_events[0].run_id == "run-1"

    @pytest.mark.asyncio
    async def test_load_cached(self, skills_dir: Path) -> None:
        events = []
        async def on_event(event: SkillEvent):
            events.append(event)

        manager = SkillManager(skills_directory=skills_dir, on_event=on_event)
        await manager.load("skill-a")
        await manager.load("skill-a")  # Second load should be cached

        loaded_events = [e for e in events if e.event_type == SkillEventType.LOADED]
        assert len(loaded_events) == 2
        assert loaded_events[1].metadata.get("cached") is True

    @pytest.mark.asyncio
    async def test_activate_skill(self, skills_dir: Path) -> None:
        events = []
        async def on_event(event: SkillEvent):
            events.append(event)

        manager = SkillManager(skills_directory=skills_dir, on_event=on_event)
        skill = await manager.activate("skill-a", run_id="run-1")

        assert skill.name == "skill-a"
        assert "skill-a" in manager.active_skills
        assert "skill-a" in manager.loaded_skills

        activated_events = [e for e in events if e.event_type == SkillEventType.ACTIVATED]
        assert len(activated_events) == 1

    @pytest.mark.asyncio
    async def test_deactivate_skill(self, skills_dir: Path) -> None:
        events = []
        async def on_event(event: SkillEvent):
            events.append(event)

        manager = SkillManager(skills_directory=skills_dir, on_event=on_event)
        await manager.activate("skill-a")
        await manager.deactivate("skill-a", run_id="run-1")

        assert "skill-a" not in manager.active_skills

        deactivated_events = [e for e in events if e.event_type == SkillEventType.DEACTIVATED]
        assert len(deactivated_events) == 1
        assert deactivated_events[0].run_id == "run-1"

    @pytest.mark.asyncio
    async def test_mark_used(self, skills_dir: Path) -> None:
        events = []
        async def on_event(event: SkillEvent):
            events.append(event)

        manager = SkillManager(skills_directory=skills_dir, on_event=on_event)
        await manager.activate("skill-a")
        manager.mark_used("skill-a", run_id="run-1")

        assert "skill-a" in manager.used_skills

        # Wait for async event
        await asyncio.sleep(0.01)
        
        used_events = [e for e in events if e.event_type == SkillEventType.USED]
        assert len(used_events) == 1
        assert used_events[0].run_id == "run-1"

    @pytest.mark.asyncio
    async def test_mark_used_not_active(self, skills_dir: Path) -> None:
        events = []
        async def on_event(event: SkillEvent):
            events.append(event)

        manager = SkillManager(skills_directory=skills_dir, on_event=on_event)
        # Not activated, just loaded
        await manager.load("skill-a")
        manager.mark_used("skill-a", run_id="run-1")

        assert "skill-a" not in manager.used_skills
        
        used_events = [e for e in events if e.event_type == SkillEventType.USED]
        assert len(used_events) == 0

    @pytest.mark.asyncio
    async def test_get_active_instructions(self, skills_dir: Path) -> None:
        manager = SkillManager(skills_directory=skills_dir)
        await manager.activate("skill-a")
        await manager.activate("skill-b")

        instructions = manager.get_active_instructions()
        assert "skill-a" in instructions
        assert "skill-b" in instructions
        assert "Instructions for skill-a" in instructions
        assert "Instructions for skill-b" in instructions

    @pytest.mark.asyncio
    async def test_clear_run_state(self, skills_dir: Path) -> None:
        manager = SkillManager(skills_directory=skills_dir)
        await manager.activate("skill-a")
        await manager.activate("skill-b")
        manager.mark_used("skill-a")
        manager.mark_used("skill-b")

        manager.clear_run_state()

        assert manager.active_skills == ()
        assert manager.loaded_skills == ()
        assert manager.used_skills == ()


class TestSkillEvents:
    def test_event_creation(self) -> None:
        event = SkillEvent(
            event_type=SkillEventType.LOADED,
            skill_name="test",
            skill_version="1.0.0",
            run_id="run-1",
            parent_run_id="parent-1",
            metadata={"key": "value"},
        )

        assert event.event_type == SkillEventType.LOADED
        assert event.skill_name == "test"
        assert event.skill_version == "1.0.0"
        assert event.run_id == "run-1"
        assert event.parent_run_id == "parent-1"
        assert event.metadata["key"] == "value"

    def test_event_to_dict(self) -> None:
        event = SkillEvent(
            event_type=SkillEventType.ACTIVATED,
            skill_name="test",
        )

        d = event.to_dict()
        assert d["event_type"] == "skill.activated"
        assert d["skill_name"] == "test"
        assert "timestamp" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])