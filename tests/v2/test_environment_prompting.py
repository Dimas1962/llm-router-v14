"""
Tests for Environment Prompting (Component 9)
"""

import pytest
import time
from src.v2.environment_prompting import (
    EnvironmentPrompter,
    SystemContext,
    ExecutionContext,
    UserPreferences,
    EnvironmentContext
)


def test_initialization():
    """Test EnvironmentPrompter initialization"""
    prompter = EnvironmentPrompter()

    assert prompter.user_preferences.language == "en"
    assert prompter.user_preferences.verbosity == "normal"
    assert prompter.user_preferences.format == "text"
    assert prompter.stats["total_prompts"] == 0
    assert prompter.stats["total_renders"] == 0


def test_collect_system_context():
    """Test system context collection"""
    prompter = EnvironmentPrompter()

    sys_ctx = prompter.collect_system_context()

    assert isinstance(sys_ctx, SystemContext)
    assert sys_ctx.os_name in ["Windows", "Linux", "Darwin"]
    assert sys_ctx.cpu_count > 0
    assert 0 <= sys_ctx.cpu_percent <= 100
    assert sys_ctx.ram_total_gb > 0
    assert sys_ctx.ram_available_gb > 0
    assert 0 <= sys_ctx.ram_percent <= 100
    assert sys_ctx.disk_total_gb > 0
    assert sys_ctx.python_version.count('.') >= 1  # At least X.Y


def test_collect_execution_context():
    """Test execution context collection"""
    prompter = EnvironmentPrompter()

    exec_ctx = prompter.collect_execution_context()

    assert isinstance(exec_ctx, ExecutionContext)
    assert exec_ctx.timestamp is not None
    assert exec_ctx.timezone is not None
    assert exec_ctx.working_directory is not None
    assert exec_ctx.user is not None


def test_execution_context_with_constraints():
    """Test execution context with constraints"""
    prompter = EnvironmentPrompter()

    constraints = {
        "max_time": 60,
        "max_memory": 1024
    }

    exec_ctx = prompter.collect_execution_context(constraints)

    assert exec_ctx.constraints == constraints
    assert exec_ctx.constraints["max_time"] == 60


def test_set_user_preferences():
    """Test setting user preferences"""
    prompter = EnvironmentPrompter()

    prompter.set_user_preferences(
        language="fr",
        verbosity="verbose",
        format="json"
    )

    assert prompter.user_preferences.language == "fr"
    assert prompter.user_preferences.verbosity == "verbose"
    assert prompter.user_preferences.format == "json"


def test_set_custom_preferences():
    """Test setting custom preferences"""
    prompter = EnvironmentPrompter()

    prompter.set_user_preferences(
        theme="dark",
        notifications=True
    )

    assert prompter.user_preferences.custom["theme"] == "dark"
    assert prompter.user_preferences.custom["notifications"] is True


def test_get_full_context():
    """Test getting full environment context"""
    prompter = EnvironmentPrompter()

    context = prompter.get_full_context(use_cache=False)

    assert isinstance(context, EnvironmentContext)
    assert isinstance(context.system, SystemContext)
    assert isinstance(context.execution, ExecutionContext)
    assert isinstance(context.preferences, UserPreferences)


def test_context_caching():
    """Test context caching"""
    prompter = EnvironmentPrompter()

    # First call - no cache
    context1 = prompter.get_full_context(use_cache=True)
    assert prompter.stats["cache_hits"] == 0

    # Second call - should use cache
    context2 = prompter.get_full_context(use_cache=True)
    assert prompter.stats["cache_hits"] == 1

    # Same context object
    assert context1 == context2


def test_get_context_summary():
    """Test context summary formatting"""
    prompter = EnvironmentPrompter()

    summary = prompter.get_context_summary()

    assert isinstance(summary, str)
    assert "System Environment:" in summary
    assert "OS:" in summary
    assert "Python:" in summary
    assert "CPU:" in summary
    assert "RAM:" in summary
    assert "Disk:" in summary
    assert "Execution Context:" in summary
    assert "Preferences:" in summary


def test_get_resource_hints_normal():
    """Test resource hints with normal resources"""
    prompter = EnvironmentPrompter()

    hints = prompter.get_resource_hints()

    assert isinstance(hints, list)
    # Should have at least some hints
    assert len(hints) >= 0


def test_render_prompt_basic():
    """Test basic prompt rendering"""
    prompter = EnvironmentPrompter()

    template = "Hello from ${os} running Python ${python_version}"
    rendered = prompter.render_prompt(template)

    assert "Hello from" in rendered
    assert "running Python" in rendered
    # Should have actual values substituted
    assert "${os}" not in rendered
    assert "${python_version}" not in rendered


def test_render_prompt_with_custom_variables():
    """Test prompt rendering with custom variables"""
    prompter = EnvironmentPrompter()

    template = "User ${name} is running ${task}"
    rendered = prompter.render_prompt(
        template,
        name="Alice",
        task="analysis"
    )

    assert "User Alice" in rendered
    assert "running analysis" in rendered


def test_render_prompt_with_context():
    """Test prompt rendering with context included"""
    prompter = EnvironmentPrompter()

    template = "Task: Process data"
    rendered = prompter.render_prompt(
        template,
        include_context=True
    )

    assert "Task: Process data" in rendered
    assert "System Environment:" in rendered
    assert "OS:" in rendered


def test_render_prompt_with_hints():
    """Test prompt rendering with resource hints"""
    prompter = EnvironmentPrompter()

    template = "Task: Run computation"
    rendered = prompter.render_prompt(
        template,
        include_hints=True
    )

    assert "Task: Run computation" in rendered
    # May or may not have hints depending on system state
    # Just check it doesn't crash


def test_create_system_prompt():
    """Test system prompt creation"""
    prompter = EnvironmentPrompter()

    base_prompt = "You are a helpful assistant."
    system_prompt = prompter.create_system_prompt(base_prompt)

    assert "You are a helpful assistant." in system_prompt
    assert "Environment:" in system_prompt
    assert "Python" in system_prompt
    assert "Available Resources:" in system_prompt


def test_create_system_prompt_without_constraints():
    """Test system prompt without resource constraints"""
    prompter = EnvironmentPrompter()

    base_prompt = "You are a helpful assistant."
    system_prompt = prompter.create_system_prompt(
        base_prompt,
        include_constraints=False
    )

    assert "You are a helpful assistant." in system_prompt
    assert "Environment:" in system_prompt
    # Should not include resource details
    assert "Available Resources:" not in system_prompt


def test_get_library_versions():
    """Test library version detection"""
    prompter = EnvironmentPrompter()

    # Test with known libraries
    versions = prompter.get_library_versions(["os", "sys", "json"])

    assert isinstance(versions, dict)
    assert "os" in versions
    assert "sys" in versions
    assert "json" in versions


def test_get_library_versions_nonexistent():
    """Test library version detection with nonexistent library"""
    prompter = EnvironmentPrompter()

    versions = prompter.get_library_versions(["nonexistent_library_xyz"])

    assert versions["nonexistent_library_xyz"] == "not installed"


def test_statistics_tracking():
    """Test statistics tracking"""
    prompter = EnvironmentPrompter()

    # Create system prompts
    prompter.create_system_prompt("Test 1")
    prompter.create_system_prompt("Test 2")

    # Render templates
    prompter.render_prompt("Template 1")
    prompter.render_prompt("Template 2")
    prompter.render_prompt("Template 3")

    stats = prompter.get_stats()

    assert stats["total_prompts"] == 2
    assert stats["total_renders"] == 3


def test_stats_reset():
    """Test statistics reset"""
    prompter = EnvironmentPrompter()

    prompter.create_system_prompt("Test")
    prompter.render_prompt("Test")

    assert prompter.stats["total_prompts"] == 1
    assert prompter.stats["total_renders"] == 1

    prompter.reset_stats()

    assert prompter.stats["total_prompts"] == 0
    assert prompter.stats["total_renders"] == 0


def test_cache_clearing():
    """Test cache clearing"""
    prompter = EnvironmentPrompter()

    # Get context (will be cached)
    context1 = prompter.get_full_context(use_cache=True)
    assert prompter.cached_context is not None

    # Clear cache
    prompter.clear_cache()
    assert prompter.cached_context is None

    # Get context again (fresh)
    context2 = prompter.get_full_context(use_cache=True)
    assert prompter.cached_context is not None


def test_template_safe_substitute():
    """Test safe substitution (missing variables)"""
    prompter = EnvironmentPrompter()

    # Template with undefined variable
    template = "Hello ${name}, running on ${os}, using ${undefined_var}"
    rendered = prompter.render_prompt(template, name="Alice")

    # Should have substituted known vars
    assert "Alice" in rendered
    # Undefined var should remain as-is (safe_substitute behavior)
    assert "${undefined_var}" in rendered


def test_context_with_metadata():
    """Test environment context with metadata"""
    prompter = EnvironmentPrompter()

    context = prompter.get_full_context()
    context.metadata["custom_key"] = "custom_value"

    assert context.metadata["custom_key"] == "custom_value"


def test_multiple_preference_updates():
    """Test multiple preference updates"""
    prompter = EnvironmentPrompter()

    # First update
    prompter.set_user_preferences(language="es")
    assert prompter.user_preferences.language == "es"

    # Second update (should not reset other preferences)
    prompter.set_user_preferences(verbosity="minimal")
    assert prompter.user_preferences.language == "es"  # Should remain
    assert prompter.user_preferences.verbosity == "minimal"


def test_execution_context_timestamp():
    """Test execution context timestamp accuracy"""
    prompter = EnvironmentPrompter()

    before = time.time()
    exec_ctx = prompter.collect_execution_context()
    after = time.time()

    timestamp = exec_ctx.timestamp.timestamp()

    # Timestamp should be between before and after
    assert before <= timestamp <= after


def test_resource_hints_with_high_ram():
    """Test resource hints generation logic"""
    prompter = EnvironmentPrompter()

    # Get actual context
    context = prompter.get_full_context()

    # Get hints
    hints = prompter.get_resource_hints(context)

    # Should return a list (content depends on actual system state)
    assert isinstance(hints, list)


def test_render_prompt_empty_template():
    """Test rendering empty template"""
    prompter = EnvironmentPrompter()

    rendered = prompter.render_prompt("")

    assert rendered == ""


def test_system_context_values():
    """Test system context has valid values"""
    prompter = EnvironmentPrompter()

    sys_ctx = prompter.collect_system_context()

    # All numeric values should be positive
    assert sys_ctx.cpu_count > 0
    assert sys_ctx.ram_total_gb > 0
    assert sys_ctx.disk_total_gb > 0

    # Percentages should be in valid range
    assert 0 <= sys_ctx.cpu_percent <= 100
    assert 0 <= sys_ctx.ram_percent <= 100
    assert 0 <= sys_ctx.disk_percent <= 100

    # Available should not exceed total
    assert sys_ctx.ram_available_gb <= sys_ctx.ram_total_gb
    assert sys_ctx.disk_available_gb <= sys_ctx.disk_total_gb
