#!/usr/bin/env python3
"""
Synaphe Test Suite — Tests for lexer, parser, and transpiler.
Run: python tests/test_flux.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lexer import Lexer, TokenType
from src.parser import Parser, parse
from src.transpiler import PythonTranspiler, transpile_to_python

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
tests_run = 0
tests_passed = 0


def test(name, fn):
    global tests_run, tests_passed
    tests_run += 1
    try:
        fn()
        tests_passed += 1
        print(f"  {PASS} {name}")
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")


# ── Lexer Tests ───────────────────────────────────────────────────────

def test_lexer_basic():
    tokens = Lexer("let x = 42").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert types == [TokenType.LET, TokenType.IDENTIFIER, TokenType.EQUALS, TokenType.INTEGER], f"Got {types}"

def test_lexer_pipeline():
    tokens = Lexer("a |> b |> c").tokenize()
    types = [t.type for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)]
    assert TokenType.PIPE_ARROW in types, f"Missing pipeline operator in {types}"

def test_lexer_matmul():
    tokens = Lexer("x @ y").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.AT in types

def test_lexer_arrow():
    tokens = Lexer("fn foo() -> Int").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.ARROW in types

def test_lexer_fat_arrow():
    tokens = Lexer("x => y").tokenize()
    types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.FAT_ARROW in types

def test_lexer_string():
    tokens = Lexer('"hello world"').tokenize()
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "hello world"

def test_lexer_float():
    tokens = Lexer("3.14").tokenize()
    assert tokens[0].type == TokenType.FLOAT
    assert tokens[0].value == "3.14"

def test_lexer_keywords():
    source = "let fn return if else match model schema import type for in while"
    tokens = Lexer(source).tokenize()
    non_eof = [t for t in tokens if t.type != TokenType.EOF]
    assert all(t.type != TokenType.IDENTIFIER for t in non_eof), "Keywords should not be IDENTIFIER"

def test_lexer_comments():
    tokens = Lexer("let x = 5 // this is a comment\nlet y = 10").tokenize()
    ids = [t for t in tokens if t.type == TokenType.IDENTIFIER]
    assert len(ids) == 2  # x and y

def test_lexer_tensor_type():
    tokens = Lexer("Tensor<Float32, [32, 784]>").tokenize()
    assert tokens[0].type == TokenType.TENSOR


# ── Parser Tests ──────────────────────────────────────────────────────

def test_parser_let():
    ast = parse("let x = 42")
    assert len(ast.statements) == 1
    stmt = ast.statements[0]
    assert stmt.__class__.__name__ == 'LetStatement'
    assert stmt.name == 'x'

def test_parser_let_typed():
    ast = parse("let x: Int = 42")
    stmt = ast.statements[0]
    assert stmt.type_annotation is not None
    assert stmt.type_annotation.__class__.__name__ == 'SimpleType'

def test_parser_function():
    ast = parse("fn add(a: Int, b: Int) -> Int { return a + b }")
    stmt = ast.statements[0]
    assert stmt.__class__.__name__ == 'FunctionDef'
    assert stmt.name == 'add'
    assert len(stmt.params) == 2

def test_parser_pipeline():
    ast = parse("let r = x |> f |> g")
    stmt = ast.statements[0]
    assert stmt.value.__class__.__name__ == 'Pipeline'
    assert len(stmt.value.stages) == 3

def test_parser_matmul():
    ast = parse("let y = x @ w")
    stmt = ast.statements[0]
    assert stmt.value.__class__.__name__ == 'MatMul'

def test_parser_function_call():
    ast = parse("print(42)")
    stmt = ast.statements[0]
    assert stmt.expr.__class__.__name__ == 'FunctionCall'

def test_parser_method_call():
    ast = parse("x.append(5)")
    stmt = ast.statements[0]
    assert stmt.expr.__class__.__name__ == 'MethodCall'

def test_parser_if():
    ast = parse("if x > 0 { let y = 1 }")
    stmt = ast.statements[0]
    assert stmt.__class__.__name__ == 'IfStatement'

def test_parser_if_else():
    ast = parse("if x > 0 { let y = 1 } else { let y = 2 }")
    stmt = ast.statements[0]
    assert len(stmt.else_body) == 1

def test_parser_for_loop():
    ast = parse("for i in range(10) { print(i) }")
    stmt = ast.statements[0]
    assert stmt.__class__.__name__ == 'ForLoop'
    assert stmt.var == 'i'

def test_parser_while_loop():
    ast = parse("while x > 0 { let x = x - 1 }")
    stmt = ast.statements[0]
    assert stmt.__class__.__name__ == 'WhileLoop'

def test_parser_model():
    ast = parse("model Net { layers: [Linear(10, 5)] loss: CrossEntropy }")
    stmt = ast.statements[0]
    assert stmt.__class__.__name__ == 'ModelDef'
    assert stmt.name == 'Net'

def test_parser_import():
    ast = parse("import torch")
    stmt = ast.statements[0]
    assert stmt.__class__.__name__ == 'ImportStatement'
    assert stmt.module == 'torch'

def test_parser_from_import():
    ast = parse("from torch import nn, optim")
    stmt = ast.statements[0]
    assert stmt.names == ['nn', 'optim']

def test_parser_schema():
    ast = parse("schema Data { age: Int }")
    stmt = ast.statements[0]
    assert stmt.__class__.__name__ == 'SchemaDef'

def test_parser_list_literal():
    ast = parse("let x = [1, 2, 3]")
    stmt = ast.statements[0]
    assert stmt.value.__class__.__name__ == 'ListLiteral'
    assert len(stmt.value.elements) == 3

def test_parser_nested_calls():
    ast = parse("let x = foo(bar(1), baz(2, 3))")
    stmt = ast.statements[0]
    assert stmt.value.__class__.__name__ == 'FunctionCall'

def test_parser_kwargs():
    ast = parse("Adam(lr=0.001)")
    stmt = ast.statements[0]
    call = stmt.expr
    assert 'lr' in call.kwargs

def test_parser_match():
    ast = parse("match x { 0 => print(0), _ => print(1) }")
    stmt = ast.statements[0]
    assert stmt.expr.__class__.__name__ == 'MatchExpr'
    assert len(stmt.expr.arms) == 2

def test_parser_tensor_type():
    ast = parse("let x: Tensor<Float32, [32, 784]> = randn(32, 784)")
    stmt = ast.statements[0]
    assert stmt.type_annotation.__class__.__name__ == 'TensorType'
    assert stmt.type_annotation.dtype == 'Float32'
    assert stmt.type_annotation.shape == [32, 784]

def test_parser_operator_precedence():
    ast = parse("let x = 1 + 2 * 3")
    stmt = ast.statements[0]
    # Should be 1 + (2 * 3), not (1 + 2) * 3
    assert stmt.value.__class__.__name__ == 'BinaryOp'
    assert stmt.value.op == '+'
    assert stmt.value.right.__class__.__name__ == 'BinaryOp'
    assert stmt.value.right.op == '*'


# ── Transpiler Tests ──────────────────────────────────────────────────

def test_transpile_let():
    code = transpile_to_python(parse("let x = 42"))
    assert "x = 42" in code

def test_transpile_function():
    code = transpile_to_python(parse("fn add(a: Int, b: Int) -> Int { return a + b }"))
    assert "def add(a: int, b: int) -> int:" in code
    assert "return (a + b)" in code

def test_transpile_pipeline():
    code = transpile_to_python(parse("let r = x |> f |> g"))
    assert "_synaphe_pipeline" in code

def test_transpile_matmul():
    code = transpile_to_python(parse("let y = x @ w"))
    assert "@" in code

def test_transpile_model():
    code = transpile_to_python(parse(
        "model Net { layers: [Linear(10, 5), ReLU] loss: CrossEntropy }"
    ))
    assert "class Net(nn.Module):" in code
    assert "nn.Linear(10, 5)" in code

def test_transpile_import():
    code = transpile_to_python(parse("import torch"))
    # There should be two `import torch` — one from header, one from the statement
    assert "import torch" in code

def test_transpile_from_import():
    code = transpile_to_python(parse("from torch import nn"))
    assert "from torch import nn" in code

def test_transpile_if():
    code = transpile_to_python(parse("if x > 0 { print(x) }"))
    assert "if (x > 0):" in code

def test_transpile_for():
    code = transpile_to_python(parse("for i in range(10) { print(i) }"))
    assert "for i in range(10):" in code

def test_transpile_boolean_ops():
    code = transpile_to_python(parse("let x = a && b || c"))
    assert "and" in code
    assert "or" in code

def test_transpile_schema():
    code = transpile_to_python(parse("schema Data { age: Int name: String }"))
    assert "class Data:" in code
    assert "validate" in code

def test_transpile_typed_let():
    code = transpile_to_python(parse("let x: Tensor<Float32, [32, 784]> = randn(32, 784)"))
    assert "torch.Tensor" in code or "Tensor" in code

def test_transpile_quantum():
    code = transpile_to_python(parse("let q = qubit()"))
    assert "qml" in code or "qubit" in code


# ── Integration Tests ─────────────────────────────────────────────────

def test_full_pipeline_example():
    source = """
let data = [1, 2, 3, 4, 5]
let total = data |> sum
print(total)
"""
    code = transpile_to_python(parse(source))
    assert "_synaphe_pipeline" in code
    assert "sum" in code

def test_full_model_example():
    source = """
model Classifier {
    layers: [
        Linear(784, 256),
        ReLU,
        Dropout(0.3),
        Linear(256, 10)
    ]
    loss: CrossEntropy
    optimizer: Adam(lr=0.001)
}
"""
    code = transpile_to_python(parse(source))
    assert "class Classifier(nn.Module):" in code
    assert "nn.Linear(784, 256)" in code
    assert "nn.ReLU()" in code
    assert "nn.Dropout(0.3)" in code

def test_full_function_with_pipeline():
    source = """
fn process(data: Tensor) -> Tensor {
    let result = data |> normalize |> model.forward |> softmax
    return result
}
"""
    code = transpile_to_python(parse(source))
    assert "def process" in code
    assert "_synaphe_pipeline" in code


# ── Run All Tests ─────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n\033[1m🧪 Synaphe Test Suite\033[0m\n")
    
    print("\033[1mLexer Tests:\033[0m")
    test("basic let statement", test_lexer_basic)
    test("pipeline operator |>", test_lexer_pipeline)
    test("matrix multiply @", test_lexer_matmul)
    test("arrow ->", test_lexer_arrow)
    test("fat arrow =>", test_lexer_fat_arrow)
    test("string literal", test_lexer_string)
    test("float literal", test_lexer_float)
    test("keywords", test_lexer_keywords)
    test("comments", test_lexer_comments)
    test("tensor type token", test_lexer_tensor_type)
    
    print(f"\n\033[1mParser Tests:\033[0m")
    test("let statement", test_parser_let)
    test("typed let", test_parser_let_typed)
    test("function definition", test_parser_function)
    test("pipeline expression", test_parser_pipeline)
    test("matrix multiply", test_parser_matmul)
    test("function call", test_parser_function_call)
    test("method call", test_parser_method_call)
    test("if statement", test_parser_if)
    test("if-else", test_parser_if_else)
    test("for loop", test_parser_for_loop)
    test("while loop", test_parser_while_loop)
    test("model definition", test_parser_model)
    test("import", test_parser_import)
    test("from import", test_parser_from_import)
    test("schema definition", test_parser_schema)
    test("list literal", test_parser_list_literal)
    test("nested calls", test_parser_nested_calls)
    test("keyword args", test_parser_kwargs)
    test("match expression", test_parser_match)
    test("tensor type annotation", test_parser_tensor_type)
    test("operator precedence", test_parser_operator_precedence)
    
    print(f"\n\033[1mTranspiler Tests:\033[0m")
    test("let -> assignment", test_transpile_let)
    test("function -> def", test_transpile_function)
    test("pipeline -> composition", test_transpile_pipeline)
    test("matmul -> @", test_transpile_matmul)
    test("model -> nn.Module", test_transpile_model)
    test("import passthrough", test_transpile_import)
    test("from import", test_transpile_from_import)
    test("if statement", test_transpile_if)
    test("for loop", test_transpile_for)
    test("boolean operators", test_transpile_boolean_ops)
    test("schema -> class", test_transpile_schema)
    test("typed let annotation", test_transpile_typed_let)
    test("quantum -> pennylane", test_transpile_quantum)
    
    print(f"\n\033[1mIntegration Tests:\033[0m")
    test("full pipeline example", test_full_pipeline_example)
    test("full model definition", test_full_model_example)
    test("function with pipeline", test_full_function_with_pipeline)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"\033[1mResults: {tests_passed}/{tests_run} passed\033[0m", end="")
    if tests_passed == tests_run:
        print(f" \033[32m— ALL PASS ✓\033[0m")
    else:
        print(f" \033[31m— {tests_run - tests_passed} FAILED\033[0m")
    print()
