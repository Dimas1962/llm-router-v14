"""
Tests for AST Analyzer (Component 7)
"""

import pytest
from src.v2.ast_analyzer import (
    ASTAnalyzer,
    CodeStructure,
    ComplexityMetrics,
    QualityScore,
    FunctionInfo,
    ClassInfo
)


def test_initialization():
    """Test ASTAnalyzer initialization"""
    analyzer = ASTAnalyzer()

    assert analyzer.stats["total_analyses"] == 0
    assert analyzer.stats["total_functions"] == 0
    assert analyzer.stats["total_classes"] == 0
    assert analyzer.stats["avg_complexity"] == 0.0


def test_simple_function_analysis():
    """Test analyzing simple function"""
    analyzer = ASTAnalyzer()

    code = '''
def hello():
    """Say hello"""
    return "Hello"
'''

    structure = analyzer.analyze_code(code)

    assert len(structure.functions) == 1
    assert structure.functions[0].name == "hello"
    assert structure.functions[0].docstring == "Say hello"
    assert structure.functions[0].cyclomatic_complexity >= 1


def test_function_with_arguments():
    """Test function argument extraction"""
    analyzer = ASTAnalyzer()

    code = '''
def add(a, b, c=0):
    return a + b + c
'''

    structure = analyzer.analyze_code(code)

    func = structure.functions[0]
    assert func.name == "add"
    assert "a" in func.args
    assert "b" in func.args
    assert "c" in func.args


def test_class_extraction():
    """Test class extraction"""
    analyzer = ASTAnalyzer()

    code = '''
class MyClass:
    """A test class"""

    def __init__(self):
        pass

    def method(self):
        pass
'''

    structure = analyzer.analyze_code(code)

    assert len(structure.classes) == 1
    cls = structure.classes[0]
    assert cls.name == "MyClass"
    assert cls.docstring == "A test class"
    assert "__init__" in cls.methods
    assert "method" in cls.methods


def test_class_with_inheritance():
    """Test class with base classes"""
    analyzer = ASTAnalyzer()

    code = '''
class Parent:
    pass

class Child(Parent):
    pass
'''

    structure = analyzer.analyze_code(code)

    child_class = [c for c in structure.classes if c.name == "Child"][0]
    assert "Parent" in child_class.bases


def test_import_extraction():
    """Test import extraction"""
    analyzer = ASTAnalyzer()

    code = '''
import os
import sys
from typing import List, Dict
from collections import defaultdict
'''

    structure = analyzer.analyze_code(code)

    assert len(structure.imports) == 4

    # Check regular imports
    os_import = [i for i in structure.imports if i.module == "os"][0]
    assert os_import.is_from is False

    # Check from imports
    typing_import = [i for i in structure.imports if i.module == "typing"][0]
    assert typing_import.is_from is True
    assert "List" in typing_import.names
    assert "Dict" in typing_import.names


def test_cyclomatic_complexity():
    """Test cyclomatic complexity calculation"""
    analyzer = ASTAnalyzer()

    # Simple function (complexity = 1)
    simple_code = '''
def simple():
    return 42
'''

    # Complex function with branches
    complex_code = '''
def complex(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x - y
    else:
        return 0
'''

    simple_structure = analyzer.analyze_code(simple_code)
    complex_structure = analyzer.analyze_code(complex_code)

    assert simple_structure.functions[0].cyclomatic_complexity == 1
    assert complex_structure.functions[0].cyclomatic_complexity > 1


def test_cognitive_complexity():
    """Test cognitive complexity calculation"""
    analyzer = ASTAnalyzer()

    code = '''
def nested_logic(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
    return 0
'''

    structure = analyzer.analyze_code(code)

    # Nested logic should have higher cognitive complexity
    assert structure.functions[0].cognitive_complexity >= 3


def test_line_counting():
    """Test line counting"""
    analyzer = ASTAnalyzer()

    code = '''# Comment line
def foo():
    # Another comment
    x = 1

    y = 2
    return x + y
'''

    structure = analyzer.analyze_code(code)

    assert structure.total_lines > 0
    assert structure.code_lines > 0
    assert structure.comment_lines >= 2
    assert structure.blank_lines >= 1


def test_complexity_metrics():
    """Test comprehensive complexity metrics"""
    analyzer = ASTAnalyzer()

    code = '''
def func1():
    if True:
        return 1
    return 0

def func2():
    for i in range(10):
        if i % 2 == 0:
            print(i)
'''

    metrics = analyzer.calculate_complexity_metrics(code)

    assert isinstance(metrics, ComplexityMetrics)
    assert metrics.cyclomatic_complexity > 0
    assert metrics.cognitive_complexity >= 0
    assert 0 <= metrics.maintainability_index <= 100
    assert "volume" in metrics.halstead_metrics


def test_dependency_graph():
    """Test dependency graph building"""
    analyzer = ASTAnalyzer()

    code = '''
import os
from typing import List, Dict
from collections import defaultdict
'''

    graph = analyzer.build_dependency_graph(code)

    assert "typing" in graph
    assert "List" in graph["typing"]
    assert "Dict" in graph["typing"]
    assert "collections" in graph
    assert "defaultdict" in graph["collections"]


def test_documentation_coverage():
    """Test documentation coverage calculation"""
    analyzer = ASTAnalyzer()

    # All documented
    documented_code = '''
def func1():
    """Documented function"""
    pass

class Class1:
    """Documented class"""
    pass
'''

    # Partially documented
    partial_code = '''
def func1():
    """Documented"""
    pass

def func2():
    pass
'''

    full_coverage = analyzer.calculate_documentation_coverage(documented_code)
    partial_coverage = analyzer.calculate_documentation_coverage(partial_code)

    assert full_coverage == 100.0
    assert partial_coverage == 50.0


def test_code_quality_scoring():
    """Test code quality scoring"""
    analyzer = ASTAnalyzer()

    good_code = '''
def simple_function(x):
    """Well documented simple function"""
    return x * 2

class SimpleClass:
    """Well documented simple class"""
    pass
'''

    score = analyzer.score_code_quality(good_code)

    assert isinstance(score, QualityScore)
    assert 0 <= score.overall_score <= 100
    assert 0 <= score.complexity_score <= 100
    assert 0 <= score.documentation_score <= 100
    assert 0 <= score.structure_score <= 100


def test_security_pattern_detection():
    """Test security pattern detection"""
    analyzer = ASTAnalyzer()

    # Code with eval (dangerous)
    dangerous_code = '''
def execute_code(code_str):
    result = eval(code_str)
    return result
'''

    # Code with hardcoded password
    secret_code = '''
API_KEY = "secret123456"
password = "hardcoded_password"
'''

    issues1 = analyzer.detect_security_patterns(dangerous_code)
    issues2 = analyzer.detect_security_patterns(secret_code)

    assert len(issues1) > 0
    assert any("eval" in issue.lower() for issue in issues1)

    assert len(issues2) > 0
    assert any("secret" in issue.lower() or "password" in issue.lower() for issue in issues2)


def test_async_function_detection():
    """Test async function detection"""
    analyzer = ASTAnalyzer()

    code = '''
async def async_function():
    await some_task()
    return True
'''

    structure = analyzer.analyze_code(code)

    assert len(structure.functions) == 1
    assert structure.functions[0].is_async is True


def test_statistics_tracking():
    """Test statistics tracking"""
    analyzer = ASTAnalyzer()

    code1 = '''
def func1():
    pass

def func2():
    if True:
        return 1
'''

    code2 = '''
class MyClass:
    pass
'''

    analyzer.analyze_code(code1)
    analyzer.analyze_code(code2)

    stats = analyzer.get_stats()

    assert stats["total_analyses"] == 2
    assert stats["total_functions"] == 2
    assert stats["total_classes"] == 1


def test_stats_reset():
    """Test statistics reset"""
    analyzer = ASTAnalyzer()

    analyzer.analyze_code("def foo(): pass")

    assert analyzer.stats["total_analyses"] == 1

    analyzer.reset_stats()

    assert analyzer.stats["total_analyses"] == 0
    assert analyzer.stats["total_functions"] == 0


def test_invalid_syntax():
    """Test handling invalid Python code"""
    analyzer = ASTAnalyzer()

    invalid_code = '''
def broken(:
    pass
'''

    with pytest.raises(ValueError):
        analyzer.analyze_code(invalid_code)


def test_empty_code():
    """Test analyzing empty code"""
    analyzer = ASTAnalyzer()

    code = ""

    structure = analyzer.analyze_code(code)

    assert len(structure.functions) == 0
    assert len(structure.classes) == 0
    assert structure.total_lines == 1  # Empty string has 1 line


def test_complex_nested_structure():
    """Test complex nested class and function structure"""
    analyzer = ASTAnalyzer()

    code = '''
class OuterClass:
    """Outer class"""

    class InnerClass:
        """Inner class"""

        def inner_method(self):
            """Inner method"""
            def nested_function():
                return 42
            return nested_function()

    def outer_method(self):
        """Outer method"""
        pass
'''

    structure = analyzer.analyze_code(code)

    # Should find both outer and inner classes
    assert len(structure.classes) >= 2

    # Should find all functions/methods
    assert len(structure.functions) >= 3


def test_return_type_annotation():
    """Test extraction of return type annotations"""
    analyzer = ASTAnalyzer()

    code = '''
def typed_function(x: int) -> str:
    return str(x)
'''

    structure = analyzer.analyze_code(code)

    func = structure.functions[0]
    assert func.returns is not None
    assert "str" in func.returns


def test_quality_score_with_issues():
    """Test quality score detects issues"""
    analyzer = ASTAnalyzer()

    # Complex function with no documentation
    problematic_code = '''
def complex_function(a, b, c, d, e):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        return True
    return False
'''

    score = analyzer.score_code_quality(problematic_code)

    # Should have issues due to high complexity and no documentation
    assert len(score.issues) > 0
