"""
Synaphe — A programming language for hybrid AI and quantum computing workflows.
Transpiles to Python with compile-time tensor shape checking.
"""

__version__ = "0.3.0"

from .lexer import Lexer, Token, TokenType, LexerError
from .parser import Parser, parse, ParseError
from .ast_nodes import *
from .transpiler import PythonTranspiler, transpile_to_python
