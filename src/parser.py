"""
Synaphe Parser — Recursive descent parser that transforms tokens into an AST.
Handles operator precedence, pipeline chains, tensor types, and all Synaphe syntax.
"""

from typing import List, Optional, Callable
from .lexer import Token, TokenType, Lexer
from .ast_nodes import *


class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        self.token = token
        super().__init__(f"Parse error at L{token.line}:{token.col}: {message} (got {token.type.name}: {token.value!r})")


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = [t for t in tokens if t.type != TokenType.NEWLINE]
        self.pos = 0
    
    # ── Utilities ─────────────────────────────────────────────────────
    
    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF
    
    def peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]
    
    def advance(self) -> Token:
        tok = self.current()
        self.pos += 1
        return tok
    
    def expect(self, type: TokenType, value: str = None) -> Token:
        tok = self.current()
        if tok.type != type:
            raise ParseError(f"Expected {type.name}" + (f" '{value}'" if value else ""), tok)
        if value and tok.value != value:
            raise ParseError(f"Expected '{value}'", tok)
        return self.advance()
    
    def match(self, *types: TokenType) -> Optional[Token]:
        if self.current().type in types:
            return self.advance()
        return None
    
    def check(self, type: TokenType, value: str = None) -> bool:
        tok = self.current()
        if tok.type != type:
            return False
        if value and tok.value != value:
            return False
        return True
    
    # ── Top-Level ─────────────────────────────────────────────────────
    
    def parse(self) -> Program:
        stmts = []
        while not self.check(TokenType.EOF):
            stmts.append(self.parse_statement())
        return Program(statements=stmts)
    
    # ── Statements ────────────────────────────────────────────────────
    
    def parse_statement(self) -> Statement:
        tok = self.current()
        
        if tok.type == TokenType.LET:
            return self.parse_let()
        elif tok.type == TokenType.FN:
            return self.parse_function_def()
        elif tok.type == TokenType.IF:
            return self.parse_if_statement()
        elif tok.type == TokenType.FOR:
            return self.parse_for_loop()
        elif tok.type == TokenType.WHILE:
            return self.parse_while_loop()
        elif tok.type == TokenType.RETURN:
            return self.parse_return()
        elif tok.type == TokenType.IMPORT:
            return self.parse_import()
        elif tok.type == TokenType.FROM:
            return self.parse_from_import()
        elif tok.type == TokenType.MODEL:
            # Lookahead: model Name { = definition; model.xxx = expression
            if self.peek(1).type == TokenType.IDENTIFIER:
                return self.parse_model_def()
            else:
                expr = self.parse_expression()
                return ExprStatement(expr=expr, line=tok.line, col=tok.col)
        elif tok.type == TokenType.SCHEMA:
            return self.parse_schema_def()
        elif tok.type == TokenType.TYPE:
            return self.parse_type_alias()
        elif tok.type == TokenType.AT:
            return self.parse_decorated()
        else:
            expr = self.parse_expression()
            # Check for assignment: expr = value (reassignment)
            if self.check(TokenType.EQUALS):
                self.advance()
                value = self.parse_expression()
                return LetStatement(
                    name=self._expr_to_assign_target(expr),
                    value=value, mutable=True,
                    line=tok.line, col=tok.col
                )
            return ExprStatement(expr=expr, line=tok.line, col=tok.col)
    
    def _expr_to_assign_target(self, expr: Expression) -> str:
        """Convert an expression to an assignment target string."""
        if isinstance(expr, Identifier):
            return expr.name
        if isinstance(expr, MemberAccess):
            obj = self._expr_to_assign_target(expr.object)
            return f"{obj}.{expr.member}"
        if isinstance(expr, IndexAccess):
            obj = self._expr_to_assign_target(expr.object)
            idx_code = str(expr.index.value) if hasattr(expr.index, 'value') else '...'
            return f"{obj}[{idx_code}]"
        return str(expr)
    
    def parse_let(self) -> LetStatement:
        tok = self.expect(TokenType.LET)
        # Allow 'model' and 'type' keywords as variable names
        name_tok = self.current()
        if name_tok.type in (TokenType.IDENTIFIER, TokenType.MODEL, TokenType.TYPE):
            self.advance()
        else:
            raise ParseError("Expected variable name", name_tok)
        
        type_ann = None
        if self.match(TokenType.COLON):
            type_ann = self.parse_type_annotation()
        
        self.expect(TokenType.EQUALS)
        value = self.parse_expression()
        
        return LetStatement(
            name=name_tok.value, type_annotation=type_ann,
            value=value, line=tok.line, col=tok.col
        )
    
    def parse_function_def(self, decorators=None) -> FunctionDef:
        tok = self.expect(TokenType.FN)
        name_tok = self.expect(TokenType.IDENTIFIER)
        
        self.expect(TokenType.LPAREN)
        params = self.parse_params()
        self.expect(TokenType.RPAREN)
        
        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.parse_type_annotation()
        
        self.expect(TokenType.LBRACE)
        body = self.parse_block()
        self.expect(TokenType.RBRACE)
        
        return FunctionDef(
            name=name_tok.value, params=params, return_type=return_type,
            body=body, decorators=decorators or [],
            line=tok.line, col=tok.col
        )
    
    def parse_params(self) -> List[Param]:
        params = []
        # Keywords that can also be used as parameter names
        param_name_types = {TokenType.IDENTIFIER, TokenType.MODEL, TokenType.TYPE}
        while not self.check(TokenType.RPAREN):
            tok = self.current()
            if tok.type not in param_name_types:
                raise ParseError(f"Expected parameter name", tok)
            name_tok = self.advance()
            type_ann = None
            default = None
            
            if self.match(TokenType.COLON):
                type_ann = self.parse_type_annotation()
            if self.match(TokenType.EQUALS):
                default = self.parse_expression()
            
            params.append(Param(
                name=name_tok.value, type_annotation=type_ann,
                default=default, line=name_tok.line, col=name_tok.col
            ))
            
            if not self.check(TokenType.RPAREN):
                self.expect(TokenType.COMMA)
        return params
    
    def parse_block(self) -> List[Statement]:
        stmts = []
        while not self.check(TokenType.RBRACE) and not self.check(TokenType.EOF):
            stmts.append(self.parse_statement())
        return stmts
    
    def parse_if_statement(self) -> IfStatement:
        tok = self.expect(TokenType.IF)
        cond = self.parse_expression()
        self.expect(TokenType.LBRACE)
        then_body = self.parse_block()
        self.expect(TokenType.RBRACE)
        
        else_body = []
        if self.match(TokenType.ELSE):
            if self.check(TokenType.IF):
                else_body = [self.parse_if_statement()]
            else:
                self.expect(TokenType.LBRACE)
                else_body = self.parse_block()
                self.expect(TokenType.RBRACE)
        
        return IfStatement(
            condition=cond, then_body=then_body, else_body=else_body,
            line=tok.line, col=tok.col
        )
    
    def parse_for_loop(self) -> ForLoop:
        tok = self.expect(TokenType.FOR)
        var_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.IN)
        iterable = self.parse_expression()
        self.expect(TokenType.LBRACE)
        body = self.parse_block()
        self.expect(TokenType.RBRACE)
        return ForLoop(var=var_tok.value, iterable=iterable, body=body, line=tok.line, col=tok.col)
    
    def parse_while_loop(self) -> WhileLoop:
        tok = self.expect(TokenType.WHILE)
        cond = self.parse_expression()
        self.expect(TokenType.LBRACE)
        body = self.parse_block()
        self.expect(TokenType.RBRACE)
        return WhileLoop(condition=cond, body=body, line=tok.line, col=tok.col)
    
    def parse_return(self) -> ReturnStatement:
        tok = self.expect(TokenType.RETURN)
        value = None
        if not self.check(TokenType.RBRACE) and not self.check(TokenType.EOF):
            value = self.parse_expression()
        return ReturnStatement(value=value, line=tok.line, col=tok.col)
    
    def parse_import(self) -> ImportStatement:
        tok = self.expect(TokenType.IMPORT)
        module_tok = self.expect(TokenType.IDENTIFIER)
        module_parts = [module_tok.value]
        while self.match(TokenType.DOT):
            module_parts.append(self.expect(TokenType.IDENTIFIER).value)
        
        alias = None
        # Check for 'as' alias (identifier with value 'as')
        if self.check(TokenType.IDENTIFIER) and self.current().value == 'as':
            self.advance()
            alias = self.expect(TokenType.IDENTIFIER).value
        
        return ImportStatement(
            module='.'.join(module_parts), alias=alias,
            line=tok.line, col=tok.col
        )
    
    def parse_from_import(self) -> ImportStatement:
        tok = self.expect(TokenType.FROM)
        module_tok = self.expect(TokenType.IDENTIFIER)
        module_parts = [module_tok.value]
        while self.match(TokenType.DOT):
            module_parts.append(self.expect(TokenType.IDENTIFIER).value)
        
        self.expect(TokenType.IMPORT)
        names = [self.expect(TokenType.IDENTIFIER).value]
        while self.match(TokenType.COMMA):
            names.append(self.expect(TokenType.IDENTIFIER).value)
        
        return ImportStatement(
            module='.'.join(module_parts), names=names,
            line=tok.line, col=tok.col
        )
    
    def parse_model_def(self) -> ModelDef:
        tok = self.expect(TokenType.MODEL)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LBRACE)
        
        fields = {}
        while not self.check(TokenType.RBRACE):
            field_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            
            if self.check(TokenType.LBRACKET):
                # List of layers
                self.advance()
                layers = []
                while not self.check(TokenType.RBRACKET):
                    layers.append(self.parse_expression())
                    if not self.check(TokenType.RBRACKET):
                        self.match(TokenType.COMMA)
                self.expect(TokenType.RBRACKET)
                fields[field_name] = layers
            else:
                fields[field_name] = self.parse_expression()
        
        self.expect(TokenType.RBRACE)
        return ModelDef(name=name_tok.value, fields=fields, line=tok.line, col=tok.col)
    
    def parse_schema_def(self) -> SchemaDef:
        tok = self.expect(TokenType.SCHEMA)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LBRACE)
        
        fields = []
        while not self.check(TokenType.RBRACE):
            fname = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            ftype = self.parse_type_annotation()
            
            constraint = None
            if self.match(TokenType.WHERE):
                constraint = self.parse_expression()
            
            fields.append(SchemaField(name=fname, type_annotation=ftype, constraint=constraint))
        
        self.expect(TokenType.RBRACE)
        return SchemaDef(name=name_tok.value, fields=fields, line=tok.line, col=tok.col)
    
    def parse_type_alias(self) -> TypeAlias:
        tok = self.expect(TokenType.TYPE)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.EQUALS)
        defn = self.parse_type_annotation()
        return TypeAlias(name=name_tok.value, definition=defn, line=tok.line, col=tok.col)
    
    def parse_decorated(self) -> Statement:
        decorators = []
        while self.check(TokenType.AT):
            self.advance()
            name = self.expect(TokenType.IDENTIFIER).value
            args = []
            if self.match(TokenType.LPAREN):
                while not self.check(TokenType.RPAREN):
                    args.append(self.parse_expression())
                    if not self.check(TokenType.RPAREN):
                        self.match(TokenType.COMMA)
                self.expect(TokenType.RPAREN)
            decorators.append(f"@{name}({', '.join(str(a) for a in args)})" if args else f"@{name}")
        
        if self.check(TokenType.FN):
            return self.parse_function_def(decorators=decorators)
        else:
            raise ParseError("Expected function definition after decorator", self.current())
    
    # ── Type Annotations ──────────────────────────────────────────────
    
    def parse_type_annotation(self) -> TypeAnnotation:
        tok = self.current()
        
        if tok.type == TokenType.TENSOR:
            return self.parse_tensor_type()
        
        type_name_types = {
            TokenType.INT, TokenType.FLOAT_TYPE, TokenType.FLOAT32,
            TokenType.FLOAT64, TokenType.STRING_TYPE, TokenType.BOOL_TYPE,
            TokenType.IDENTIFIER, TokenType.PROB
        }
        
        if tok.type in type_name_types:
            name = self.advance().value
            
            # Generic type: Name<T>
            if self.check(TokenType.LT):
                self.advance()
                params = [self.parse_type_annotation()]
                while self.match(TokenType.COMMA):
                    params.append(self.parse_type_annotation())
                self.expect(TokenType.GT)
                return GenericType(name=name, params=params, line=tok.line, col=tok.col)
            
            return SimpleType(name=name, line=tok.line, col=tok.col)
        
        raise ParseError("Expected type annotation", tok)
    
    def parse_tensor_type(self) -> TensorType:
        tok = self.expect(TokenType.TENSOR)
        dtype = "Float32"
        shape = []
        
        if self.match(TokenType.LT):
            # Read dtype
            dtype_tok = self.advance()
            dtype = dtype_tok.value
            
            if self.match(TokenType.COMMA):
                # Read shape: [dim1, dim2, ...]
                self.expect(TokenType.LBRACKET)
                while not self.check(TokenType.RBRACKET):
                    if self.check(TokenType.INTEGER):
                        shape.append(int(self.advance().value))
                    elif self.check(TokenType.IDENTIFIER):
                        shape.append(self.advance().value)  # symbolic dim
                    if not self.check(TokenType.RBRACKET):
                        self.match(TokenType.COMMA)
                self.expect(TokenType.RBRACKET)
            
            self.expect(TokenType.GT)
        
        return TensorType(dtype=dtype, shape=shape, line=tok.line, col=tok.col)
    
    # ── Expressions (Precedence Climbing) ─────────────────────────────
    
    def parse_expression(self) -> Expression:
        return self.parse_pipeline()
    
    def parse_pipeline(self) -> Expression:
        """Lowest precedence: pipeline |> operator"""
        expr = self.parse_or()
        
        if self.check(TokenType.PIPE_ARROW):
            stages = [expr]
            while self.match(TokenType.PIPE_ARROW):
                stages.append(self.parse_or())
            return Pipeline(stages=stages, line=expr.line, col=expr.col)
        
        return expr
    
    def parse_or(self) -> Expression:
        left = self.parse_and()
        while self.check(TokenType.OR):
            op = self.advance().value
            right = self.parse_and()
            left = BinaryOp(op=op, left=left, right=right, line=left.line, col=left.col)
        return left
    
    def parse_and(self) -> Expression:
        left = self.parse_equality()
        while self.check(TokenType.AND):
            op = self.advance().value
            right = self.parse_equality()
            left = BinaryOp(op=op, left=left, right=right, line=left.line, col=left.col)
        return left
    
    def parse_equality(self) -> Expression:
        left = self.parse_comparison()
        while self.current().type in (TokenType.EQ, TokenType.NEQ):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinaryOp(op=op, left=left, right=right, line=left.line, col=left.col)
        return left
    
    def parse_comparison(self) -> Expression:
        left = self.parse_addition()
        while self.current().type in (TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            op = self.advance().value
            right = self.parse_addition()
            left = BinaryOp(op=op, left=left, right=right, line=left.line, col=left.col)
        return left
    
    def parse_addition(self) -> Expression:
        left = self.parse_multiplication()
        while self.current().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplication()
            left = BinaryOp(op=op, left=left, right=right, line=left.line, col=left.col)
        return left
    
    def parse_multiplication(self) -> Expression:
        left = self.parse_matmul()
        while self.current().type in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.advance().value
            right = self.parse_matmul()
            left = BinaryOp(op=op, left=left, right=right, line=left.line, col=left.col)
        return left
    
    def parse_matmul(self) -> Expression:
        left = self.parse_unary()
        while self.check(TokenType.AT):
            self.advance()
            right = self.parse_unary()
            left = MatMul(left=left, right=right, line=left.line, col=left.col)
        return left
    
    def parse_unary(self) -> Expression:
        if self.current().type in (TokenType.MINUS, TokenType.NOT):
            tok = self.advance()
            operand = self.parse_unary()
            return UnaryOp(op=tok.value, operand=operand, line=tok.line, col=tok.col)
        return self.parse_postfix()
    
    def parse_postfix(self) -> Expression:
        expr = self.parse_primary()
        
        while True:
            if self.check(TokenType.LPAREN):
                # Function call
                self.advance()
                args, kwargs = self.parse_call_args()
                self.expect(TokenType.RPAREN)
                expr = FunctionCall(callee=expr, args=args, kwargs=kwargs, line=expr.line, col=expr.col)
            elif self.check(TokenType.DOT):
                self.advance()
                member = self.expect(TokenType.IDENTIFIER).value
                if self.check(TokenType.LPAREN):
                    self.advance()
                    args, _ = self.parse_call_args()
                    self.expect(TokenType.RPAREN)
                    expr = MethodCall(object=expr, method=member, args=args, line=expr.line, col=expr.col)
                else:
                    expr = MemberAccess(object=expr, member=member, line=expr.line, col=expr.col)
            elif self.check(TokenType.LBRACKET):
                self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexAccess(object=expr, index=index, line=expr.line, col=expr.col)
            else:
                break
        
        return expr
    
    def parse_call_args(self):
        args = []
        kwargs = {}
        while not self.check(TokenType.RPAREN):
            # Check for keyword arg: name=value
            if (self.check(TokenType.IDENTIFIER) and 
                self.peek(1).type == TokenType.EQUALS):
                key = self.advance().value
                self.advance()  # skip =
                kwargs[key] = self.parse_expression()
            else:
                args.append(self.parse_expression())
            
            if not self.check(TokenType.RPAREN):
                self.expect(TokenType.COMMA)
        return args, kwargs
    
    def parse_primary(self) -> Expression:
        tok = self.current()
        
        if tok.type == TokenType.INTEGER:
            self.advance()
            return IntLiteral(value=int(tok.value), line=tok.line, col=tok.col)
        
        if tok.type == TokenType.FLOAT:
            self.advance()
            return FloatLiteral(value=float(tok.value), line=tok.line, col=tok.col)
        
        if tok.type == TokenType.STRING:
            self.advance()
            return StringLiteral(value=tok.value, line=tok.line, col=tok.col)
        
        if tok.type == TokenType.BOOLEAN:
            self.advance()
            return BoolLiteral(value=(tok.value == 'true'), line=tok.line, col=tok.col)
        
        if tok.type == TokenType.IDENTIFIER:
            self.advance()
            return Identifier(name=tok.value, line=tok.line, col=tok.col)
        
        if tok.type in (TokenType.QUBIT, TokenType.QREGISTER, TokenType.MEASURE, TokenType.MODEL):
            self.advance()
            return Identifier(name=tok.value, line=tok.line, col=tok.col)
        
        if tok.type == TokenType.MATCH:
            return self.parse_match_expr()
        
        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        if tok.type == TokenType.LBRACKET:
            return self.parse_list_literal()
        
        if tok.type == TokenType.LBRACE:
            return self.parse_dict_literal()
        
        raise ParseError(f"Unexpected token in expression", tok)
    
    def parse_list_literal(self) -> ListLiteral:
        tok = self.expect(TokenType.LBRACKET)
        elements = []
        while not self.check(TokenType.RBRACKET):
            elements.append(self.parse_expression())
            if not self.check(TokenType.RBRACKET):
                self.expect(TokenType.COMMA)
        self.expect(TokenType.RBRACKET)
        return ListLiteral(elements=elements, line=tok.line, col=tok.col)
    
    def parse_dict_literal(self) -> DictLiteral:
        tok = self.expect(TokenType.LBRACE)
        pairs = []
        while not self.check(TokenType.RBRACE):
            key = self.parse_expression()
            self.expect(TokenType.COLON)
            value = self.parse_expression()
            pairs.append((key, value))
            if not self.check(TokenType.RBRACE):
                self.expect(TokenType.COMMA)
        self.expect(TokenType.RBRACE)
        return DictLiteral(pairs=pairs, line=tok.line, col=tok.col)
    
    def parse_match_expr(self) -> MatchExpr:
        tok = self.expect(TokenType.MATCH)
        subject = self.parse_expression()
        self.expect(TokenType.LBRACE)
        
        arms = []
        while not self.check(TokenType.RBRACE):
            pattern = self.parse_pattern()
            self.expect(TokenType.FAT_ARROW)
            body = self.parse_expression()
            arms.append(MatchArm(pattern=pattern, body=body))
            # Optional comma between arms
            self.match(TokenType.COMMA)
        
        self.expect(TokenType.RBRACE)
        return MatchExpr(subject=subject, arms=arms, line=tok.line, col=tok.col)
    
    def parse_pattern(self) -> Pattern:
        tok = self.current()
        
        if tok.type == TokenType.IDENTIFIER and tok.value == '_':
            self.advance()
            return WildcardPattern(line=tok.line, col=tok.col)
        
        if tok.type == TokenType.INTEGER:
            self.advance()
            return LiteralPattern(value=int(tok.value), line=tok.line, col=tok.col)
        
        if tok.type == TokenType.STRING:
            self.advance()
            return LiteralPattern(value=tok.value, line=tok.line, col=tok.col)
        
        if tok.type == TokenType.IDENTIFIER:
            name = self.advance().value
            if self.match(TokenType.LPAREN):
                args = []
                while not self.check(TokenType.RPAREN):
                    args.append(self.parse_pattern())
                    if not self.check(TokenType.RPAREN):
                        self.match(TokenType.COMMA)
                self.expect(TokenType.RPAREN)
                return ConstructorPattern(name=name, args=args, line=tok.line, col=tok.col)
            return IdentifierPattern(name=name, line=tok.line, col=tok.col)
        
        if tok.type == TokenType.LBRACKET:
            self.advance()
            elems = []
            while not self.check(TokenType.RBRACKET):
                elems.append(self.parse_pattern())
                if not self.check(TokenType.RBRACKET):
                    self.match(TokenType.COMMA)
            self.expect(TokenType.RBRACKET)
            return ListPattern(elements=elems, line=tok.line, col=tok.col)
        
        raise ParseError("Expected pattern", tok)


def parse(source: str) -> Program:
    """Convenience function: source code -> AST"""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
