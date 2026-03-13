"""
Synaphe Lexer — Tokenizes .synaphe source code into a stream of tokens.
Handles all Flux syntax including pipeline operators, tensor types,
quantum keywords, and standard operators.
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


class TokenType(Enum):
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    
    # Identifiers & Keywords
    IDENTIFIER = auto()
    LET = auto()
    FN = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    MATCH = auto()
    MODEL = auto()
    SCHEMA = auto()
    IMPORT = auto()
    FROM = auto()
    TYPE = auto()
    WHERE = auto()
    FOR = auto()
    IN = auto()
    WHILE = auto()
    
    # Quantum keywords
    QUBIT = auto()
    QREGISTER = auto()
    MEASURE = auto()
    
    # Type keywords
    INT = auto()
    FLOAT_TYPE = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    STRING_TYPE = auto()
    BOOL_TYPE = auto()
    TENSOR = auto()
    PROB = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    AT = auto()           # @ — matrix multiply / decorators
    PIPE_ARROW = auto()   # |> — pipeline operator
    PIPE = auto()         # | — alternative/union
    ARROW = auto()        # -> — return type
    FAT_ARROW = auto()    # => — match arm
    EQUALS = auto()       # =
    EQ = auto()           # ==
    NEQ = auto()          # !=
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    AND = auto()          # &&
    OR = auto()           # ||
    NOT = auto()          # !
    DOT = auto()
    DOTDOT = auto()       # .. range
    COLON = auto()
    DOUBLE_COLON = auto() # :: path separator
    COMMA = auto()
    SEMICOLON = auto()
    HASH = auto()         # # for comments
    UNDERSCORE = auto()   # _ wildcard
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    
    # Special
    NEWLINE = auto()
    EOF = auto()
    DECORATOR = auto()    # @target, @differentiable etc.


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"


# Keywords mapping
KEYWORDS = {
    'let': TokenType.LET,
    'fn': TokenType.FN,
    'return': TokenType.RETURN,
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'match': TokenType.MATCH,
    'model': TokenType.MODEL,
    'schema': TokenType.SCHEMA,
    'import': TokenType.IMPORT,
    'from': TokenType.FROM,
    'type': TokenType.TYPE,
    'where': TokenType.WHERE,
    'for': TokenType.FOR,
    'in': TokenType.IN,
    'while': TokenType.WHILE,
    'true': TokenType.BOOLEAN,
    'false': TokenType.BOOLEAN,
    'qubit': TokenType.QUBIT,
    'qregister': TokenType.QREGISTER,
    'measure': TokenType.MEASURE,
    'Int': TokenType.INT,
    'Float': TokenType.FLOAT_TYPE,
    'Float32': TokenType.FLOAT32,
    'Float64': TokenType.FLOAT64,
    'String': TokenType.STRING_TYPE,
    'Bool': TokenType.BOOL_TYPE,
    'Tensor': TokenType.TENSOR,
    'Prob': TokenType.PROB,
}


class LexerError(Exception):
    def __init__(self, message: str, line: int, col: int):
        self.line = line
        self.col = col
        super().__init__(f"Lexer error at L{line}:{col}: {message}")


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []
    
    def peek(self, offset: int = 0) -> Optional[str]:
        idx = self.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return None
    
    def advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch
    
    def add_token(self, type: TokenType, value: str, line: int, col: int):
        self.tokens.append(Token(type, value, line, col))
    
    def skip_whitespace(self):
        while self.pos < len(self.source) and self.source[self.pos] in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        # Single-line comment: // ...
        if self.pos + 1 < len(self.source) and self.source[self.pos:self.pos+2] == '//':
            while self.pos < len(self.source) and self.source[self.pos] != '\n':
                self.advance()
            return True
        # Block comment: /* ... */
        if self.pos + 1 < len(self.source) and self.source[self.pos:self.pos+2] == '/*':
            self.advance()  # /
            self.advance()  # *
            while self.pos + 1 < len(self.source):
                if self.source[self.pos:self.pos+2] == '*/':
                    self.advance()
                    self.advance()
                    return True
                self.advance()
            raise LexerError("Unterminated block comment", self.line, self.col)
        return False
    
    def read_string(self) -> Token:
        line, col = self.line, self.col
        quote = self.advance()  # consume opening quote
        result = []
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch == '\\':
                self.advance()
                if self.pos < len(self.source):
                    escape = self.advance()
                    escape_map = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}
                    result.append(escape_map.get(escape, '\\' + escape))
            elif ch == quote:
                self.advance()
                return Token(TokenType.STRING, ''.join(result), line, col)
            elif ch == '\n':
                raise LexerError("Unterminated string", line, col)
            else:
                result.append(self.advance())
        raise LexerError("Unterminated string", line, col)
    
    def read_number(self) -> Token:
        line, col = self.line, self.col
        result = []
        is_float = False
        
        while self.pos < len(self.source) and (self.source[self.pos].isdigit() or self.source[self.pos] in '._'):
            ch = self.source[self.pos]
            if ch == '.':
                if self.peek(1) == '.':  # range operator
                    break
                if is_float:
                    break
                is_float = True
            elif ch == '_':
                self.advance()
                continue
            result.append(self.advance())
        
        value = ''.join(result)
        if is_float:
            return Token(TokenType.FLOAT, value, line, col)
        return Token(TokenType.INTEGER, value, line, col)
    
    def read_identifier(self) -> Token:
        line, col = self.line, self.col
        result = []
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
            result.append(self.advance())
        
        value = ''.join(result)
        token_type = KEYWORDS.get(value, TokenType.IDENTIFIER)
        return Token(token_type, value, line, col)
    
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            self.skip_whitespace()
            
            if self.pos >= len(self.source):
                break
            
            if self.skip_comment():
                continue
            
            ch = self.source[self.pos]
            line, col = self.line, self.col
            
            # Newlines (significant for statement separation)
            if ch == '\n':
                self.advance()
                self.add_token(TokenType.NEWLINE, '\\n', line, col)
                continue
            
            # Strings
            if ch in '"\'':
                self.tokens.append(self.read_string())
                continue
            
            # Numbers
            if ch.isdigit() or (ch == '.' and self.peek(1) and self.peek(1).isdigit()):
                self.tokens.append(self.read_number())
                continue
            
            # Identifiers and keywords
            if ch.isalpha() or ch == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Multi-character operators
            two_char = self.source[self.pos:self.pos+2] if self.pos + 1 < len(self.source) else ''
            
            if two_char == '|>':
                self.advance(); self.advance()
                self.add_token(TokenType.PIPE_ARROW, '|>', line, col)
            elif two_char == '->':
                self.advance(); self.advance()
                self.add_token(TokenType.ARROW, '->', line, col)
            elif two_char == '=>':
                self.advance(); self.advance()
                self.add_token(TokenType.FAT_ARROW, '=>', line, col)
            elif two_char == '==':
                self.advance(); self.advance()
                self.add_token(TokenType.EQ, '==', line, col)
            elif two_char == '!=':
                self.advance(); self.advance()
                self.add_token(TokenType.NEQ, '!=', line, col)
            elif two_char == '<=':
                self.advance(); self.advance()
                self.add_token(TokenType.LTE, '<=', line, col)
            elif two_char == '>=':
                self.advance(); self.advance()
                self.add_token(TokenType.GTE, '>=', line, col)
            elif two_char == '&&':
                self.advance(); self.advance()
                self.add_token(TokenType.AND, '&&', line, col)
            elif two_char == '||':
                self.advance(); self.advance()
                self.add_token(TokenType.OR, '||', line, col)
            elif two_char == '::':
                self.advance(); self.advance()
                self.add_token(TokenType.DOUBLE_COLON, '::', line, col)
            elif two_char == '..':
                self.advance(); self.advance()
                self.add_token(TokenType.DOTDOT, '..', line, col)
            
            # Single-character operators
            elif ch == '+':
                self.advance()
                self.add_token(TokenType.PLUS, '+', line, col)
            elif ch == '-':
                self.advance()
                self.add_token(TokenType.MINUS, '-', line, col)
            elif ch == '*':
                self.advance()
                self.add_token(TokenType.STAR, '*', line, col)
            elif ch == '/':
                self.advance()
                self.add_token(TokenType.SLASH, '/', line, col)
            elif ch == '%':
                self.advance()
                self.add_token(TokenType.PERCENT, '%', line, col)
            elif ch == '@':
                self.advance()
                self.add_token(TokenType.AT, '@', line, col)
            elif ch == '|':
                self.advance()
                self.add_token(TokenType.PIPE, '|', line, col)
            elif ch == '=':
                self.advance()
                self.add_token(TokenType.EQUALS, '=', line, col)
            elif ch == '<':
                self.advance()
                self.add_token(TokenType.LT, '<', line, col)
            elif ch == '>':
                self.advance()
                self.add_token(TokenType.GT, '>', line, col)
            elif ch == '!':
                self.advance()
                self.add_token(TokenType.NOT, '!', line, col)
            elif ch == '.':
                self.advance()
                self.add_token(TokenType.DOT, '.', line, col)
            elif ch == ':':
                self.advance()
                self.add_token(TokenType.COLON, ':', line, col)
            elif ch == ',':
                self.advance()
                self.add_token(TokenType.COMMA, ',', line, col)
            elif ch == ';':
                self.advance()
                self.add_token(TokenType.SEMICOLON, ';', line, col)
            elif ch == '(':
                self.advance()
                self.add_token(TokenType.LPAREN, '(', line, col)
            elif ch == ')':
                self.advance()
                self.add_token(TokenType.RPAREN, ')', line, col)
            elif ch == '[':
                self.advance()
                self.add_token(TokenType.LBRACKET, '[', line, col)
            elif ch == ']':
                self.advance()
                self.add_token(TokenType.RBRACKET, ']', line, col)
            elif ch == '{':
                self.advance()
                self.add_token(TokenType.LBRACE, '{', line, col)
            elif ch == '}':
                self.advance()
                self.add_token(TokenType.RBRACE, '}', line, col)
            else:
                raise LexerError(f"Unexpected character: {ch!r}", line, col)
        
        self.add_token(TokenType.EOF, '', self.line, self.col)
        return self.tokens
