"""
Environment Prompting - Component 9
System context collection and prompt enhancement
"""

import os
import sys
import platform
import psutil
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from string import Template


@dataclass
class SystemContext:
    """System information context"""
    os_name: str
    os_version: str
    platform: str
    cpu_count: int
    cpu_percent: float
    ram_total_gb: float
    ram_available_gb: float
    ram_percent: float
    disk_total_gb: float
    disk_available_gb: float
    disk_percent: float
    python_version: str


@dataclass
class ExecutionContext:
    """Execution environment context"""
    timestamp: datetime
    timezone: str
    working_directory: str
    user: str
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreferences:
    """User preferences"""
    language: str = "en"
    verbosity: str = "normal"  # minimal, normal, verbose
    format: str = "text"  # text, json, markdown
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentContext:
    """Complete environment context"""
    system: SystemContext
    execution: ExecutionContext
    preferences: UserPreferences
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnvironmentPrompter:
    """
    Environment-Aware Prompt System

    Features:
    - Collect system context (OS, CPU, RAM, disk)
    - Include Python/library versions in prompts
    - Resource availability hints
    - User preference integration
    - Execution context (time, location, constraints)
    - Template rendering with environment variables
    """

    def __init__(self):
        """Initialize Environment Prompter"""
        self.user_preferences = UserPreferences()
        self.cached_context: Optional[EnvironmentContext] = None
        self.cache_ttl = 60  # seconds

        # Statistics
        self.stats = {
            "total_prompts": 0,
            "total_renders": 0,
            "cache_hits": 0
        }

    def collect_system_context(self) -> SystemContext:
        """
        Collect system information using psutil

        Returns:
            SystemContext with system details
        """
        # OS info
        os_name = platform.system()
        os_version = platform.release()
        platform_info = platform.platform()

        # CPU info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # RAM info
        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024 ** 3)
        ram_available_gb = ram.available / (1024 ** 3)
        ram_percent = ram.percent

        # Disk info
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024 ** 3)
        disk_available_gb = disk.free / (1024 ** 3)
        disk_percent = disk.percent

        # Python version
        python_version = sys.version.split()[0]

        return SystemContext(
            os_name=os_name,
            os_version=os_version,
            platform=platform_info,
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            ram_percent=ram_percent,
            disk_total_gb=disk_total_gb,
            disk_available_gb=disk_available_gb,
            disk_percent=disk_percent,
            python_version=python_version
        )

    def collect_execution_context(
        self,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ExecutionContext:
        """
        Collect execution context

        Args:
            constraints: Optional execution constraints

        Returns:
            ExecutionContext with execution details
        """
        timestamp = datetime.now()
        timezone = datetime.now().astimezone().tzname()
        working_directory = os.getcwd()
        user = os.getenv('USER', os.getenv('USERNAME', 'unknown'))

        return ExecutionContext(
            timestamp=timestamp,
            timezone=timezone,
            working_directory=working_directory,
            user=user,
            constraints=constraints or {}
        )

    def set_user_preferences(
        self,
        language: Optional[str] = None,
        verbosity: Optional[str] = None,
        format: Optional[str] = None,
        **custom
    ):
        """
        Set user preferences

        Args:
            language: Preferred language
            verbosity: Verbosity level
            format: Output format
            **custom: Custom preferences
        """
        if language:
            self.user_preferences.language = language
        if verbosity:
            self.user_preferences.verbosity = verbosity
        if format:
            self.user_preferences.format = format
        if custom:
            self.user_preferences.custom.update(custom)

    def get_full_context(
        self,
        use_cache: bool = True,
        constraints: Optional[Dict[str, Any]] = None
    ) -> EnvironmentContext:
        """
        Get complete environment context

        Args:
            use_cache: Whether to use cached context
            constraints: Optional execution constraints

        Returns:
            EnvironmentContext
        """
        # Check cache
        if use_cache and self.cached_context:
            exec_ctx = self.collect_execution_context(constraints)
            if (exec_ctx.timestamp - self.cached_context.execution.timestamp).seconds < self.cache_ttl:
                self.stats["cache_hits"] += 1
                return self.cached_context

        # Collect fresh context
        system_ctx = self.collect_system_context()
        exec_ctx = self.collect_execution_context(constraints)

        context = EnvironmentContext(
            system=system_ctx,
            execution=exec_ctx,
            preferences=self.user_preferences
        )

        self.cached_context = context
        return context

    def get_context_summary(self, context: Optional[EnvironmentContext] = None) -> str:
        """
        Get formatted context summary

        Args:
            context: Optional context to summarize (uses cached if None)

        Returns:
            Formatted context string
        """
        if context is None:
            context = self.get_full_context()

        sys_ctx = context.system
        exec_ctx = context.execution
        prefs = context.preferences

        summary = f"""System Environment:
- OS: {sys_ctx.os_name} {sys_ctx.os_version}
- Python: {sys_ctx.python_version}
- CPU: {sys_ctx.cpu_count} cores ({sys_ctx.cpu_percent}% used)
- RAM: {sys_ctx.ram_available_gb:.1f}/{sys_ctx.ram_total_gb:.1f} GB available ({sys_ctx.ram_percent}% used)
- Disk: {sys_ctx.disk_available_gb:.1f}/{sys_ctx.disk_total_gb:.1f} GB available ({sys_ctx.disk_percent}% used)

Execution Context:
- Time: {exec_ctx.timestamp.strftime('%Y-%m-%d %H:%M:%S')} {exec_ctx.timezone}
- User: {exec_ctx.user}
- Directory: {exec_ctx.working_directory}

Preferences:
- Language: {prefs.language}
- Verbosity: {prefs.verbosity}
- Format: {prefs.format}
"""
        return summary

    def get_resource_hints(self, context: Optional[EnvironmentContext] = None) -> List[str]:
        """
        Get resource availability hints

        Args:
            context: Optional context (uses cached if None)

        Returns:
            List of resource hints
        """
        if context is None:
            context = self.get_full_context()

        hints = []
        sys_ctx = context.system

        # RAM hints
        if sys_ctx.ram_percent > 90:
            hints.append("⚠️ Very low RAM available - consider memory-efficient operations")
        elif sys_ctx.ram_percent > 75:
            hints.append("⚠️ RAM usage high - monitor memory consumption")
        elif sys_ctx.ram_available_gb > 8:
            hints.append("✓ Sufficient RAM available for large operations")

        # CPU hints
        if sys_ctx.cpu_percent > 80:
            hints.append("⚠️ High CPU usage - consider async or parallel operations carefully")
        elif sys_ctx.cpu_count >= 8:
            hints.append(f"✓ {sys_ctx.cpu_count} CPU cores available for parallel processing")

        # Disk hints
        if sys_ctx.disk_percent > 90:
            hints.append("⚠️ Very low disk space - avoid large file operations")
        elif sys_ctx.disk_percent > 75:
            hints.append("⚠️ Disk space running low")

        return hints

    def render_prompt(
        self,
        template: str,
        include_context: bool = False,
        include_hints: bool = False,
        **variables
    ) -> str:
        """
        Render prompt template with environment variables

        Args:
            template: Prompt template with ${var} placeholders
            include_context: Whether to include system context
            include_hints: Whether to include resource hints
            **variables: Additional variables to substitute

        Returns:
            Rendered prompt string
        """
        context = self.get_full_context()

        # Build variable dictionary
        env_vars = {
            'os': context.system.os_name,
            'os_version': context.system.os_version,
            'python_version': context.system.python_version,
            'cpu_count': context.system.cpu_count,
            'ram_available': f"{context.system.ram_available_gb:.1f}GB",
            'user': context.execution.user,
            'time': context.execution.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'language': context.preferences.language,
            'verbosity': context.preferences.verbosity,
            'format': context.preferences.format
        }

        # Add custom variables
        env_vars.update(variables)

        # Render template
        tmpl = Template(template)
        rendered = tmpl.safe_substitute(env_vars)

        # Add context if requested
        if include_context:
            rendered = self.get_context_summary(context) + "\n" + rendered

        # Add hints if requested
        if include_hints:
            hints = self.get_resource_hints(context)
            if hints:
                rendered = rendered + "\n\nResource Hints:\n" + "\n".join(hints)

        # Update stats
        self.stats["total_renders"] += 1

        return rendered

    def create_system_prompt(
        self,
        base_prompt: str,
        include_constraints: bool = True
    ) -> str:
        """
        Create system prompt with environment context

        Args:
            base_prompt: Base system prompt
            include_constraints: Whether to include resource constraints

        Returns:
            Enhanced system prompt
        """
        context = self.get_full_context()
        sys_ctx = context.system

        enhanced = base_prompt + "\n\n"
        enhanced += f"Environment: {sys_ctx.os_name}, Python {sys_ctx.python_version}\n"

        if include_constraints:
            enhanced += f"Available Resources: {sys_ctx.ram_available_gb:.1f}GB RAM, "
            enhanced += f"{sys_ctx.cpu_count} CPU cores\n"

            # Add constraints
            if sys_ctx.ram_percent > 75:
                enhanced += "Note: Limited memory available - prioritize efficiency\n"
            if sys_ctx.cpu_percent > 80:
                enhanced += "Note: High CPU load - avoid intensive computations\n"

        self.stats["total_prompts"] += 1

        return enhanced

    def get_library_versions(self, libraries: List[str]) -> Dict[str, str]:
        """
        Get versions of installed libraries

        Args:
            libraries: List of library names

        Returns:
            Dictionary mapping library names to versions
        """
        versions = {}

        for lib in libraries:
            try:
                module = __import__(lib)
                version = getattr(module, '__version__', 'unknown')
                versions[lib] = version
            except ImportError:
                versions[lib] = 'not installed'
            except Exception:
                versions[lib] = 'error'

        return versions

    def get_stats(self) -> Dict[str, Any]:
        """Get prompter statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_prompts": 0,
            "total_renders": 0,
            "cache_hits": 0
        }

    def clear_cache(self):
        """Clear cached context"""
        self.cached_context = None
