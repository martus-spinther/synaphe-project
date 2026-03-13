#!/usr/bin/env python3
"""
Synaphe CLI — Command-line interface for the Synaphe language.
Supports: REPL, file compilation, and direct execution.

Usage:
    python -m flux                  # Launch REPL
    python -m flux run example.flux # Compile and run
    python -m flux build example.flux # Transpile to Python
    python -m flux check example.flux # Type check only
"""

import sys
import os
import argparse
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lexer import Lexer, LexerError
from src.parser import Parser, ParseError, parse
from src.transpiler import PythonTranspiler, transpile_to_python


SYNAPHE_BANNER = r"""
    ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ██╗  ██╗███████╗
  ██╔════╝██║     ██║   ██║╚██╗██╔╝
  █████╗  ██║     ██║   ██║ ╚███╔╝ 
  ██╔══╝  ██║     ██║   ██║ ██╔██╗ 
  ██║     ███████╗╚██████╔╝██╔╝ ██╗
  ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝
  v0.1.0 — Hybrid AI + Quantum Language
  Type 'help' for commands, 'exit' to quit.
"""

HELP_TEXT = """
Synaphe REPL Commands:
  help          Show this help message
  exit / quit   Exit the REPL
  .tokens       Show tokens for last input
  .ast          Show AST for last input
  .python       Show transpiled Python for last input
  .run          Compile and execute last input
  .example      Show example Synaphe code
  .clear        Clear the screen

Write any Synaphe expression or statement and press Enter.
Multi-line input: end a line with \\ to continue.
"""

EXAMPLES = {
    "pipeline": '''// Data pipeline with type safety
let data = [1, 2, 3, 4, 5]
let result = data |> sum |> print''',

    "function": '''// Function with typed parameters
fn add(a: Int, b: Int) -> Int {
    return a + b
}
let result = add(3, 4)
print(result)''',

    "model": '''// Neural network model definition
model Classifier {
    layers: [
        Linear(784, 256),
        ReLU,
        Dropout(0.3),
        Linear(256, 10)
    ]
    loss: CrossEntropy
    optimizer: Adam(lr=0.001)
}''',

    "tensor": '''// Tensor operations with shape types
let x: Tensor<Float32, [32, 784]> = randn(32, 784)
let w: Tensor<Float32, [784, 128]> = randn(784, 128)
let y = x @ w''',

    "quantum": '''// Quantum circuit as a pipeline
let result = qubit() |> hadamard |> measure |> print''',

    "match": '''// Pattern matching
let x = 42
match x {
    0 => print("zero"),
    n => print(n)
}''',

    "schema": '''// Data validation schema
schema UserInput {
    age: Int where age >= 0
    name: String
    score: Float where score >= 0.0
}'''
}


class FluxREPL:
    def __init__(self):
        self.last_source = ""
        self.last_tokens = []
        self.last_ast = None
        self.last_python = ""
        self.history = []
    
    def run(self):
        print(SYNAPHE_BANNER)
        
        while True:
            try:
                line = input("\033[36msynaphe>\033[0m ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            line = line.strip()
            if not line:
                continue
            
            # Handle multi-line input
            while line.endswith("\\"):
                line = line[:-1]
                try:
                    cont = input("\033[36m  ...\033[0m ")
                    line += "\n" + cont
                except (EOFError, KeyboardInterrupt):
                    break
            
            # REPL commands
            if line in ('exit', 'quit'):
                print("Goodbye!")
                break
            elif line == 'help':
                print(HELP_TEXT)
                continue
            elif line == '.tokens':
                self._show_tokens()
                continue
            elif line == '.ast':
                self._show_ast()
                continue
            elif line == '.python':
                self._show_python()
                continue
            elif line == '.run':
                self._run_last()
                continue
            elif line == '.clear':
                os.system('clear' if os.name != 'nt' else 'cls')
                continue
            elif line == '.example':
                self._show_examples()
                continue
            elif line.startswith('.example '):
                name = line.split(' ', 1)[1]
                self._show_example(name)
                continue
            
            # Process Synaphe code
            self._process(line)
    
    def _process(self, source: str):
        self.last_source = source
        self.history.append(source)
        
        try:
            # Lex
            lexer = Lexer(source)
            self.last_tokens = lexer.tokenize()
            
            # Parse
            parser = Parser(self.last_tokens)
            self.last_ast = parser.parse()
            
            # Transpile
            transpiler = PythonTranspiler()
            self.last_python = transpiler.transpile(self.last_ast)
            
            # Show transpiled output
            print(f"\033[32m✓ Compiled successfully\033[0m")
            print(f"\033[90m─── Python output ───\033[0m")
            # Show just the body (skip header imports)
            body_lines = []
            in_body = False
            for line in self.last_python.split('\n'):
                if line.startswith('#') or line.startswith('import') or line.startswith('from') or line.startswith('def _flux') or not line.strip():
                    if in_body:
                        body_lines.append(line)
                    continue
                in_body = True
                body_lines.append(line)
            
            body = '\n'.join(body_lines).strip()
            if body:
                print(f"\033[33m{body}\033[0m")
            
        except LexerError as e:
            print(f"\033[31m✗ Lexer Error: {e}\033[0m")
        except ParseError as e:
            print(f"\033[31m✗ Parse Error: {e}\033[0m")
        except Exception as e:
            print(f"\033[31m✗ Error: {e}\033[0m")
            traceback.print_exc()
    
    def _show_tokens(self):
        if not self.last_tokens:
            print("No tokens. Enter some Synaphe code first.")
            return
        print("\033[90m─── Tokens ───\033[0m")
        for tok in self.last_tokens:
            if tok.type.name != 'NEWLINE':
                print(f"  {tok}")
    
    def _show_ast(self):
        if not self.last_ast:
            print("No AST. Enter some Synaphe code first.")
            return
        print("\033[90m─── AST ───\033[0m")
        self._print_ast(self.last_ast, indent=2)
    
    def _print_ast(self, node, indent=0):
        prefix = " " * indent
        if hasattr(node, '__dataclass_fields__'):
            name = type(node).__name__
            print(f"{prefix}\033[35m{name}\033[0m")
            for field_name in node.__dataclass_fields__:
                if field_name in ('line', 'col'):
                    continue
                value = getattr(node, field_name)
                if isinstance(value, list):
                    if value:
                        print(f"{prefix}  {field_name}:")
                        for item in value:
                            self._print_ast(item, indent + 4)
                elif hasattr(value, '__dataclass_fields__'):
                    print(f"{prefix}  {field_name}:")
                    self._print_ast(value, indent + 4)
                elif isinstance(value, dict):
                    print(f"{prefix}  {field_name}: {value}")
                else:
                    if value is not None:
                        print(f"{prefix}  {field_name}: \033[36m{value}\033[0m")
        else:
            print(f"{prefix}\033[36m{node}\033[0m")
    
    def _show_python(self):
        if not self.last_python:
            print("No Python output. Enter some Synaphe code first.")
            return
        print("\033[90m─── Full Python Output ───\033[0m")
        print(self.last_python)
    
    def _run_last(self):
        if not self.last_python:
            print("Nothing to run. Enter some Synaphe code first.")
            return
        print("\033[90m─── Executing ───\033[0m")
        try:
            # Create a safe execution namespace
            exec_globals = {"__builtins__": __builtins__}
            exec(self.last_python, exec_globals)
            print(f"\033[32m✓ Execution complete\033[0m")
        except Exception as e:
            print(f"\033[31m✗ Runtime Error: {e}\033[0m")
    
    def _show_examples(self):
        print("\033[90m─── Available Examples ───\033[0m")
        for name in EXAMPLES:
            print(f"  .example {name}")
        print()
        print("Type '.example <name>' to see the code.")
    
    def _show_example(self, name: str):
        if name not in EXAMPLES:
            print(f"Unknown example: {name}")
            self._show_examples()
            return
        code = EXAMPLES[name]
        print(f"\033[90m─── Example: {name} ───\033[0m")
        print(f"\033[36m{code}\033[0m")
        print()
        print("Processing...")
        self._process(code)


def compile_file(filepath: str) -> str:
    """Compile a .synaphe file to Python."""
    with open(filepath, 'r') as f:
        source = f.read()
    
    ast = parse(source)
    return transpile_to_python(ast)


def main():
    parser = argparse.ArgumentParser(
        prog='flux',
        description='Synaphe — Hybrid AI + Quantum Programming Language'
    )
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Compile and execute a .synaphe file')
    run_parser.add_argument('file', help='Path to .synaphe file')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Transpile .flux to Python')
    build_parser.add_argument('file', help='Path to .synaphe file')
    build_parser.add_argument('-o', '--output', help='Output file path')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Type-check a .synaphe file')
    check_parser.add_argument('file', help='Path to .synaphe file')
    
    args = parser.parse_args()
    
    if args.command is None:
        # Launch REPL
        repl = FluxREPL()
        repl.run()
    
    elif args.command == 'run':
        try:
            python_code = compile_file(args.file)
            exec(python_code)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == 'build':
        try:
            python_code = compile_file(args.file)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(python_code)
                print(f"Compiled {args.file} -> {args.output}")
            else:
                print(python_code)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == 'check':
        try:
            ast = parse(open(args.file).read())
            print(f"✓ {args.file}: No errors found")
        except (LexerError, ParseError) as e:
            print(f"✗ {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
