from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.tool import Tool, ToolResult, ToolError, ToolPolicy, ToolContext
from agent_workflow.core.entities.models.skills.skill import Skill


SKILLS_READ = "skills.read"
SKILLS_WRITE = "skills.write"


@dataclass(slots=True, frozen=True)
class SkillsListInput:
    """No arguments. Present so the tool has a real schema."""


@dataclass(slots=True, frozen=True)
class SkillsLoadInput:
    name: str


def create_skills_list_tool() -> Tool[SkillsListInput, ToolResult]:
    """List available skills."""
    
    async def handler(input_data: SkillsListInput, context: ToolContext) -> ToolResult:
        del input_data
        agent = getattr(context, "agent", None)
        if not agent or not hasattr(agent, "skill_manager") or not agent.skill_manager:
            return ToolResult(
                error=ToolError(
                    message="Skill system not available",
                    code="skills_not_available",
                    retryable=False,
                )
            )
        
        try:
            skills = await agent.skill_manager.list_available_metadata()
            
            if not skills:
                return ToolResult(output="No skills available")
            
            lines = ["Available skills:"]
            for skill in skills:
                status = "AVAILABLE"
                lines.append(f"  {skill['name']} (v{skill['version']}) - {status}")
                if skill['description']:
                    lines.append(f"    {skill['description']}")
            
            return ToolResult(output="\n".join(lines))
        
        except Exception as exc:
            return ToolResult(
                error=ToolError(
                    message=f"Failed to list skills: {exc}",
                    code="list_skills_failed",
                    retryable=True,
                )
            )
    
    return Tool(
        name="skills.list",
        description="List all available skills with their descriptions",
        input_type=SkillsListInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset([SKILLS_READ]),
            timeout=30.0,
            max_output_size=10000,
        ),
    )


def create_skills_load_tool() -> Tool[SkillsLoadInput, ToolResult]:
    """Load and activate a skill by name."""
    
    async def handler(input_data: SkillsLoadInput, context: ToolContext) -> ToolResult:
        agent = getattr(context, "agent", None)
        if not agent or not hasattr(agent, "skill_manager") or not agent.skill_manager:
            return ToolResult(
                error=ToolError(
                    message="Skill system not available",
                    code="skills_not_available",
                    retryable=False,
                )
            )
        
        skill_name = input_data.name
        if not skill_name:
            return ToolResult(
                error=ToolError(
                    message="Skill name is required",
                    code="missing_skill_name",
                    retryable=False,
                )
            )
        
        try:
            skill = await agent.load_skill(skill_name)
            
            return ToolResult(
                output=f"Loaded skill: {skill.name} v{skill.version}\n{skill.summary()}"
            )
        
        except Exception as exc:
            return ToolResult(
                error=ToolError(
                    message=f"Failed to load skill '{skill_name}': {exc}",
                    code="load_skill_failed",
                    retryable=True,
                )
            )
    
    return Tool(
        name="skills.load",
        description="Load and activate a skill by name. The skill's instructions become available for the current task.",
        input_type=SkillsLoadInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset([SKILLS_WRITE]),
            timeout=30.0,
            max_output_size=5000,
        ),
    )


def create_skills_tools() -> list[Tool]:
    """Create all skills management tools."""
    return [
        create_skills_list_tool(),
        create_skills_load_tool(),
    ]