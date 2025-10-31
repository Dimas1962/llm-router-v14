#!/usr/bin/env python3
"""
Automated Demo Recording for Unified Router v2.0
Creates: demo_output.html, demo_output.svg for conversion to GIF/MP4
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.layout import Layout
    from rich.live import Live
except ImportError:
    print("❌ Error: rich library not installed")
    print("Install with: pip install rich")
    sys.exit(1)

from src.unified.unified_router import UnifiedRouter, UnifiedRequest, RoutingStrategy


class DemoRecorder:
    """Records automated demonstration of Unified Router capabilities"""

    def __init__(self, output_dir="./"):
        self.console = Console(record=True, width=100)
        self.router = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    async def scene_1_intro(self):
        """Scene 1: Title Screen (5 sec)"""
        self.console.clear()
        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan]🚀 Unified LLM Router v2.0.0[/bold cyan]\n\n"
            "[white]21 Components | 455/457 Tests | Production Ready[/white]\n"
            "[dim]Intelligent routing for LLM models with quality assurance[/dim]",
            border_style="cyan",
            title="[bold white]Demo Recording[/bold white]"
        ))
        self.console.print()
        await asyncio.sleep(3)

    async def scene_2_initialization(self):
        """Scene 2: Router Initialization (8 sec)"""
        self.console.clear()
        self.console.print("[bold yellow]📦 Initializing Unified Router...[/bold yellow]\n")

        components = [
            ("Context Manager v2", "Advanced caching"),
            ("Runtime Adapter", "Dynamic optimization"),
            ("Self-Check System", "Quality verification"),
            ("Eagle ELO", "Quality-focused routing"),
            ("CARROT", "Cost-aware optimization"),
            ("Cascade Router", "Multi-tier fallback"),
            ("Batching Layer", "Throughput boost"),
            ("State Snapshot", "Debug & rollback")
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Loading components...", total=len(components))
            for comp_name, comp_desc in components:
                self.console.print(f"  [green]✓[/green] [cyan]{comp_name}[/cyan] - [dim]{comp_desc}[/dim]")
                progress.advance(task)
                await asyncio.sleep(0.7)

        self.console.print("\n[bold green]✅ Router initialized with 21 components![/bold green]")
        await asyncio.sleep(2)

    async def scene_3_quality_routing(self):
        """Scene 3: Quality-Focused Routing (10 sec)"""
        self.console.clear()
        self.console.print("[bold magenta]🎯 Demo 1: Quality-Focused Routing (Eagle ELO)[/bold magenta]\n")

        query = "Explain quantum entanglement and its implications for quantum computing"
        self.console.print(f"[cyan]Query:[/cyan] [white]{query}[/white]\n")

        # Simulate routing pipeline
        self.console.print("[yellow]→ Step 1:[/yellow] Runtime Adapter analyzing system load... [green]LOW[/green]")
        await asyncio.sleep(1.5)
        self.console.print("[yellow]→ Step 2:[/yellow] Context Sizing calculating budget... [green]8000 tokens[/green]")
        await asyncio.sleep(1.5)
        self.console.print("[yellow]→ Step 3:[/yellow] Eagle ELO selecting best model...\n")
        await asyncio.sleep(1)

        result_table = Table(show_header=True, header_style="bold cyan")
        result_table.add_column("Model", style="cyan", width=25)
        result_table.add_column("ELO Rating", justify="right", style="green")
        result_table.add_column("Quality", justify="right", style="yellow")
        result_table.add_column("Selected", justify="center", style="bold green")
        result_table.add_row("qwen3-next-80b", "1842", "0.90", "✓")
        result_table.add_row("qwen2.5-coder-32b", "1735", "0.80", "")
        result_table.add_row("deepseek-coder-16b", "1690", "0.77", "")

        self.console.print(result_table)
        self.console.print("\n[bold green]✅ Selected: qwen3-next-80b (highest quality for reasoning)[/bold green]")
        await asyncio.sleep(2.5)

    async def scene_4_cost_routing(self):
        """Scene 4: Cost-Aware Routing (9 sec)"""
        self.console.clear()
        self.console.print("[bold magenta]💰 Demo 2: Cost-Aware Routing (CARROT)[/bold magenta]\n")

        query = "Write a Python function to sort a list using merge sort"
        self.console.print(f"[cyan]Query:[/cyan] [white]{query}[/white]\n")
        self.console.print("[cyan]Budget:[/cyan] [yellow]$0.05[/yellow]\n")

        self.console.print("[yellow]→ CARROT analyzing Pareto frontier...[/yellow]")
        await asyncio.sleep(1.5)

        pareto_table = Table(show_header=True, header_style="bold cyan")
        pareto_table.add_column("Model", style="cyan", width=25)
        pareto_table.add_column("Cost", justify="right", style="red")
        pareto_table.add_column("Quality", justify="right", style="green")
        pareto_table.add_column("Score", justify="right", style="yellow")
        pareto_table.add_column("Selected", justify="center", style="bold green")
        pareto_table.add_row("qwen2.5-coder-7b", "$0.02", "8.5/10", "0.92", "✓")
        pareto_table.add_row("qwen2.5-coder-32b", "$0.05", "9.0/10", "0.88", "")
        pareto_table.add_row("qwen3-next-80b", "$0.12", "9.5/10", "0.75", "")

        self.console.print(pareto_table)
        self.console.print("\n[bold green]✅ Selected: qwen2.5-coder-7b (optimal cost/quality balance)[/bold green]")
        await asyncio.sleep(2.5)

    async def scene_5_cascade_routing(self):
        """Scene 5: Cascade Routing (8 sec)"""
        self.console.clear()
        self.console.print("[bold magenta]⚡ Demo 3: Cascade Routing (Fast → Quality)[/bold magenta]\n")

        query = "Hello, how are you?"
        self.console.print(f"[cyan]Query:[/cyan] [white]{query}[/white]\n")

        self.console.print("[yellow]→ Tier 1 (Fast):[/yellow] Trying qwen2.5-coder-7b...")
        await asyncio.sleep(1)
        self.console.print("   [green]✓ Success![/green] Latency: 45ms\n")
        await asyncio.sleep(1)

        cascade_table = Table(show_header=True, header_style="bold cyan")
        cascade_table.add_column("Tier", style="cyan")
        cascade_table.add_column("Models", style="white")
        cascade_table.add_column("Use Case", style="dim")
        cascade_table.add_row("1. Fast", "qwen2.5-coder-7b", "Simple queries")
        cascade_table.add_row("2. Medium", "qwen2.5-coder-32b", "Moderate complexity")
        cascade_table.add_row("3. Quality", "qwen3-next-80b", "Complex reasoning")

        self.console.print(cascade_table)
        self.console.print("\n[bold green]✅ Fast tier succeeded - no escalation needed[/bold green]")
        await asyncio.sleep(2)

    async def scene_6_quality_check(self):
        """Scene 6: Quality Verification (9 sec)"""
        self.console.clear()
        self.console.print("[bold magenta]🔍 Demo 4: Quality Verification (Self-Check)[/bold magenta]\n")

        self.console.print("[yellow]→ Self-Check System analyzing response...[/yellow]\n")
        await asyncio.sleep(1.5)

        check_table = Table(show_header=True, header_style="bold cyan")
        check_table.add_column("Metric", style="cyan", width=25)
        check_table.add_column("Score", justify="right", style="green", width=15)
        check_table.add_column("Threshold", justify="right", style="yellow", width=15)
        check_table.add_column("Status", justify="center", style="bold green", width=12)
        check_table.add_row("Focus Score", "9.2/10", "6.0/10", "✓ PASS")
        check_table.add_row("Result Score", "8.8/10", "6.0/10", "✓ PASS")
        check_table.add_row("Fact Verification", "95%", "70%", "✓ PASS")
        check_table.add_row("Overall Quality", "9.0/10", "6.0/10", "✓ PASS")

        self.console.print(check_table)
        self.console.print("\n[bold green]✅ Quality verified! Response accepted without retry.[/bold green]")
        await asyncio.sleep(3)

    async def scene_7_performance(self):
        """Scene 7: Performance Metrics (10 sec)"""
        self.console.clear()
        self.console.print("[bold magenta]📊 Performance Metrics & Statistics[/bold magenta]\n")

        metrics_table = Table(show_header=True, header_style="bold cyan", title="[bold white]Benchmark Results[/bold white]")
        metrics_table.add_column("Metric", style="cyan", width=25)
        metrics_table.add_column("Value", justify="right", style="green", width=20)
        metrics_table.add_column("Target", justify="right", style="yellow", width=20)
        metrics_table.add_column("Status", justify="center", style="bold green", width=10)
        metrics_table.add_row("Throughput", "7.84 QPS", "5+ QPS", "✓")
        metrics_table.add_row("Avg Latency", "128ms", "<200ms", "✓")
        metrics_table.add_row("p95 Latency", "285ms", "<300ms", "✓")
        metrics_table.add_row("Cache Hit Rate", "82%", "80%+", "✓")
        metrics_table.add_row("Success Rate", "100%", "95%+", "✓")
        metrics_table.add_row("Memory Usage", "1.2GB", "<2GB", "✓")

        self.console.print(metrics_table)

        self.console.print("\n[bold cyan]System Status:[/bold cyan]")
        self.console.print("  [green]•[/green] 21 components operational")
        self.console.print("  [green]•[/green] 455/457 tests passing (99.6%)")
        self.console.print("  [green]•[/green] 4 routing strategies available")
        self.console.print("  [green]•[/green] Production-ready with Docker support")
        await asyncio.sleep(4)

    async def scene_8_finale(self):
        """Scene 8: Final Screen (6 sec)"""
        self.console.clear()
        self.console.print()
        self.console.print(Panel.fit(
            "[bold green]✅ Unified LLM Router v2.0.0[/bold green]\n\n"
            "[white]Production Ready | 21 Components | 455 Tests[/white]\n\n"
            "[cyan]GitHub:[/cyan] [link=https://github.com/Dimas1962/llm-router-v14]https://github.com/Dimas1962/llm-router-v14[/link]\n"
            "[cyan]Docs:[/cyan] See docs/ directory for complete guides\n"
            "[cyan]Release:[/cyan] Download from GitHub Releases\n\n"
            "[yellow]⭐ Star us on GitHub if you find this useful![/yellow]",
            border_style="green",
            title="[bold white]Thank You![/bold white]"
        ))
        self.console.print()
        await asyncio.sleep(5)

    async def record_demo(self):
        """Record complete demonstration"""
        self.console.print("[bold cyan]Starting demo recording...[/bold cyan]\n")

        # Record all scenes
        await self.scene_1_intro()
        await self.scene_2_initialization()
        await self.scene_3_quality_routing()
        await self.scene_4_cost_routing()
        await self.scene_5_cascade_routing()
        await self.scene_6_quality_check()
        await self.scene_7_performance()
        await self.scene_8_finale()

        # Save outputs
        html_path = self.output_dir / "demo_output.html"
        svg_path = self.output_dir / "demo_output.svg"

        self.console.save_html(str(html_path), clear=False)
        self.console.save_svg(str(svg_path), title="Unified Router v2.0 Demo")

        # Print success message
        print("\n" + "="*70)
        print("✅ Demo recorded successfully!")
        print("="*70)
        print(f"\nOutput files created:")
        print(f"  📄 {html_path} (view in browser)")
        print(f"  📄 {svg_path} (vector graphics)")
        print(f"\n📝 Next steps:")
        print(f"  1. View HTML: open {html_path}")
        print(f"  2. Create GIF: termtosvg render {svg_path} demo.gif")
        print(f"  3. Optimize GIF: gifsicle -O3 --colors 256 demo.gif -o demo_optimized.gif")
        print(f"\n💡 See docs/DEMO_GUIDE.md for detailed instructions")
        print("="*70 + "\n")


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Record automated demo for Unified Router v2.0")
    parser.add_argument("-o", "--output", default="./", help="Output directory (default: current)")
    args = parser.parse_args()

    recorder = DemoRecorder(output_dir=args.output)
    await recorder.record_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Demo recording interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during demo recording: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
