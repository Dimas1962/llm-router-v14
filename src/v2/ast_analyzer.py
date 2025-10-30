"""
AST Analyzer - Component 7
Python code analysis through Abstract Syntax Tree
"""

import ast
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict


@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    lineno: int
    args: List[str]
    returns: Optional[str]
    docstring: Optional[str]
    cyclomatic_complexity: int
    cognitive_complexity: int
    is_async: bool = False


@dataclass
class ClassInfo:
    """Information about a class"""
    name: str
    lineno: int
    bases: List[str]
    methods: List[str]
    docstring: Optional[str]


@dataclass
class ImportInfo:
    """Information about imports"""
    module: str
    names: List[str]
    is_from: bool
    lineno: int


@dataclass
class CodeStructure:
    """Extracted code structure"""
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[ImportInfo]
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int


@dataclass
class ComplexityMetrics:
    """Complexity metrics"""
    cyclomatic_complexity: int
    cognitive_complexity: int
    maintainability_index: float
    halstead_metrics: Dict[str, float]


@dataclass
class QualityScore:
    """Code quality scoring"""
    overall_score: float  # 0-100
    complexity_score: float
    documentation_score: float
    structure_score: float
    issues: List[str]


class ASTAnalyzer:
    """
    AST-based Code Analyzer

    Features:
    - Parse Python code via AST
    - Calculate cyclomatic and cognitive complexity
    - Extract code structure (functions, classes, imports)
    - Build dependency graphs
    - Score code quality
    - Measure documentation coverage
    - Detect security patterns
    """

    def __init__(self):
        """Initialize AST Analyzer"""
        self.stats = {
            "total_analyses": 0,
            "total_functions": 0,
            "total_classes": 0,
            "avg_complexity": 0.0
        }
        self._total_complexity = 0.0

    def analyze_code(self, code: str) -> CodeStructure:
        """
        Analyze Python code

        Args:
            code: Python source code

        Returns:
            CodeStructure with extracted information
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")

        # Extract structure
        functions = self._extract_functions(tree, code)
        classes = self._extract_classes(tree)
        imports = self._extract_imports(tree)

        # Count lines
        lines = code.split('\n')
        total_lines = len(lines)
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        blank_lines = sum(1 for line in lines if not line.strip())
        code_lines = total_lines - comment_lines - blank_lines

        # Update stats
        self.stats["total_analyses"] += 1
        self.stats["total_functions"] += len(functions)
        self.stats["total_classes"] += len(classes)

        if functions:
            avg_func_complexity = sum(f.cyclomatic_complexity for f in functions) / len(functions)
            self._total_complexity += avg_func_complexity
            self.stats["avg_complexity"] = self._total_complexity / self.stats["total_analyses"]

        return CodeStructure(
            functions=functions,
            classes=classes,
            imports=imports,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines
        )

    def _extract_functions(self, tree: ast.AST, code: str) -> List[FunctionInfo]:
        """Extract function information"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Get arguments
                args = [arg.arg for arg in node.args.args]

                # Get return annotation
                returns = None
                if node.returns:
                    returns = ast.unparse(node.returns)

                # Get docstring
                docstring = ast.get_docstring(node)

                # Calculate complexity
                cyclomatic = self._calculate_cyclomatic_complexity(node)
                cognitive = self._calculate_cognitive_complexity(node)

                functions.append(FunctionInfo(
                    name=node.name,
                    lineno=node.lineno,
                    args=args,
                    returns=returns,
                    docstring=docstring,
                    cyclomatic_complexity=cyclomatic,
                    cognitive_complexity=cognitive,
                    is_async=isinstance(node, ast.AsyncFunctionDef)
                ))

        return functions

    def _extract_classes(self, tree: ast.AST) -> List[ClassInfo]:
        """Extract class information"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Get base classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(ast.unparse(base))

                # Get method names
                methods = [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]

                # Get docstring
                docstring = ast.get_docstring(node)

                classes.append(ClassInfo(
                    name=node.name,
                    lineno=node.lineno,
                    bases=bases,
                    methods=methods,
                    docstring=docstring
                ))

        return classes

    def _extract_imports(self, tree: ast.AST) -> List[ImportInfo]:
        """Extract import information"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.name],
                        is_from=False,
                        lineno=node.lineno
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    is_from=True,
                    lineno=node.lineno
                ))

        return imports

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        Calculate cyclomatic complexity

        Args:
            node: AST node (typically function)

        Returns:
            Cyclomatic complexity score
        """
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.Match):  # Python 3.10+
                complexity += len(child.cases)

        return complexity

    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """
        Calculate cognitive complexity

        Args:
            node: AST node

        Returns:
            Cognitive complexity score
        """
        complexity = 0
        nesting_level = 0

        def visit(n, level):
            nonlocal complexity

            # Increment for control flow structures
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + level
            elif isinstance(n, ast.ExceptHandler):
                complexity += 1 + level
            elif isinstance(n, (ast.And, ast.Or)):
                complexity += 1

            # Increment nesting for children
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                new_level = level + 1
            else:
                new_level = level

            for child in ast.iter_child_nodes(n):
                visit(child, new_level)

        for child in ast.iter_child_nodes(node):
            visit(child, nesting_level)

        return complexity

    def calculate_complexity_metrics(self, code: str) -> ComplexityMetrics:
        """
        Calculate comprehensive complexity metrics

        Args:
            code: Python source code

        Returns:
            ComplexityMetrics
        """
        structure = self.analyze_code(code)

        # Aggregate complexity
        total_cyclomatic = sum(f.cyclomatic_complexity for f in structure.functions)
        total_cognitive = sum(f.cognitive_complexity for f in structure.functions)

        # Calculate maintainability index (simplified)
        # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        # Simplified version based on complexity and lines
        if structure.code_lines > 0:
            cc_avg = total_cyclomatic / max(len(structure.functions), 1)
            maintainability = max(0, 100 - (cc_avg * 5) - (structure.code_lines / 10))
        else:
            maintainability = 100.0

        # Halstead metrics (simplified)
        halstead = {
            "volume": structure.code_lines * 4.0,
            "difficulty": total_cyclomatic / max(len(structure.functions), 1),
            "effort": structure.code_lines * total_cyclomatic
        }

        return ComplexityMetrics(
            cyclomatic_complexity=total_cyclomatic,
            cognitive_complexity=total_cognitive,
            maintainability_index=maintainability,
            halstead_metrics=halstead
        )

    def build_dependency_graph(self, code: str) -> Dict[str, List[str]]:
        """
        Build dependency graph from imports

        Args:
            code: Python source code

        Returns:
            Dictionary mapping modules to their dependencies
        """
        structure = self.analyze_code(code)

        # Build graph
        graph = defaultdict(list)

        for imp in structure.imports:
            if imp.is_from:
                # from module import name
                for name in imp.names:
                    graph[imp.module].append(name)
            else:
                # import module
                graph["__main__"].extend(imp.names)

        return dict(graph)

    def calculate_documentation_coverage(self, code: str) -> float:
        """
        Calculate documentation coverage

        Args:
            code: Python source code

        Returns:
            Coverage percentage (0-100)
        """
        structure = self.analyze_code(code)

        total_items = len(structure.functions) + len(structure.classes)
        if total_items == 0:
            return 100.0

        documented_items = 0

        for func in structure.functions:
            if func.docstring:
                documented_items += 1

        for cls in structure.classes:
            if cls.docstring:
                documented_items += 1

        return (documented_items / total_items) * 100.0

    def score_code_quality(self, code: str) -> QualityScore:
        """
        Score code quality

        Args:
            code: Python source code

        Returns:
            QualityScore with overall and component scores
        """
        structure = self.analyze_code(code)
        metrics = self.calculate_complexity_metrics(code)
        doc_coverage = self.calculate_documentation_coverage(code)

        issues = []

        # Complexity score (lower is better)
        if structure.functions:
            avg_cyclomatic = metrics.cyclomatic_complexity / len(structure.functions)
            if avg_cyclomatic > 10:
                complexity_score = max(0, 100 - (avg_cyclomatic - 10) * 5)
                issues.append(f"High average cyclomatic complexity: {avg_cyclomatic:.1f}")
            else:
                complexity_score = 100.0
        else:
            complexity_score = 100.0

        # Documentation score
        documentation_score = doc_coverage
        if doc_coverage < 50:
            issues.append(f"Low documentation coverage: {doc_coverage:.1f}%")

        # Structure score
        if structure.code_lines > 0:
            functions_per_line = len(structure.functions) / structure.code_lines * 100
            if functions_per_line < 1:
                issues.append("Very few functions for code size")
                structure_score = 70.0
            else:
                structure_score = min(100.0, functions_per_line * 10)
        else:
            structure_score = 100.0

        # Overall score (weighted average)
        overall = (
            complexity_score * 0.4 +
            documentation_score * 0.3 +
            structure_score * 0.3
        )

        return QualityScore(
            overall_score=overall,
            complexity_score=complexity_score,
            documentation_score=documentation_score,
            structure_score=structure_score,
            issues=issues
        )

    def detect_security_patterns(self, code: str) -> List[str]:
        """
        Detect potential security issues

        Args:
            code: Python source code

        Returns:
            List of detected security patterns/issues
        """
        issues = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ["Syntax error - cannot analyze"]

        # Check for dangerous patterns
        for node in ast.walk(tree):
            # eval() usage
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('eval', 'exec'):
                        issues.append(f"Dangerous function '{node.func.id}' at line {node.lineno}")
                    elif node.func.id == 'compile':
                        issues.append(f"Dynamic code compilation at line {node.lineno}")

            # SQL string concatenation (potential injection)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                if isinstance(node.left, ast.Str) or isinstance(node.right, ast.Str):
                    code_str = ast.unparse(node)
                    if any(kw in code_str.upper() for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                        issues.append(f"Potential SQL injection at line {node.lineno}")

            # Hardcoded secrets
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id.lower()
                        if any(kw in var_name for kw in ['password', 'secret', 'token', 'api_key']):
                            if isinstance(node.value, ast.Constant):
                                issues.append(f"Hardcoded secret '{target.id}' at line {node.lineno}")

        return issues

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_analyses": 0,
            "total_functions": 0,
            "total_classes": 0,
            "avg_complexity": 0.0
        }
        self._total_complexity = 0.0
