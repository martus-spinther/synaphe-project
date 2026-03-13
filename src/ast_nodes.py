"""
Synaphe AST — Abstract Syntax Tree nodes for the Synaphe language.
Every syntactic construct is represented as a dataclass node.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Any


# ── Base ──────────────────────────────────────────────────────────────

@dataclass
class ASTNode:
    line: int = 0
    col: int = 0


# ── Types ─────────────────────────────────────────────────────────────

@dataclass
class TypeAnnotation(ASTNode):
    pass

@dataclass
class SimpleType(TypeAnnotation):
    name: str = ""

@dataclass
class TensorType(TypeAnnotation):
    dtype: str = "Float32"
    shape: List[Union[str, int]] = field(default_factory=list)

@dataclass
class GenericType(TypeAnnotation):
    name: str = ""
    params: List[TypeAnnotation] = field(default_factory=list)

@dataclass
class FunctionType(TypeAnnotation):
    params: List[TypeAnnotation] = field(default_factory=list)
    return_type: Optional[TypeAnnotation] = None

@dataclass
class UnionType(TypeAnnotation):
    variants: List[TypeAnnotation] = field(default_factory=list)


# ── Expressions ───────────────────────────────────────────────────────

@dataclass
class Expression(ASTNode):
    pass

@dataclass
class IntLiteral(Expression):
    value: int = 0

@dataclass
class FloatLiteral(Expression):
    value: float = 0.0

@dataclass
class StringLiteral(Expression):
    value: str = ""

@dataclass
class BoolLiteral(Expression):
    value: bool = False

@dataclass
class Identifier(Expression):
    name: str = ""

@dataclass
class BinaryOp(Expression):
    op: str = ""
    left: Optional[Expression] = None
    right: Optional[Expression] = None

@dataclass
class UnaryOp(Expression):
    op: str = ""
    operand: Optional[Expression] = None

@dataclass
class FunctionCall(Expression):
    callee: Optional[Expression] = None
    args: List[Expression] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)

@dataclass
class MethodCall(Expression):
    object: Optional[Expression] = None
    method: str = ""
    args: List[Expression] = field(default_factory=list)

@dataclass
class MemberAccess(Expression):
    object: Optional[Expression] = None
    member: str = ""

@dataclass
class IndexAccess(Expression):
    object: Optional[Expression] = None
    index: Optional[Expression] = None

@dataclass
class Pipeline(Expression):
    """The |> pipeline operator: expr |> fn"""
    stages: List[Expression] = field(default_factory=list)

@dataclass
class MatMul(Expression):
    """The @ matrix multiplication operator"""
    left: Optional[Expression] = None
    right: Optional[Expression] = None

@dataclass
class ListLiteral(Expression):
    elements: List[Expression] = field(default_factory=list)

@dataclass
class DictLiteral(Expression):
    pairs: List[tuple] = field(default_factory=list)

@dataclass
class Lambda(Expression):
    params: List[str] = field(default_factory=list)
    body: Optional[Expression] = None

@dataclass
class IfExpr(Expression):
    condition: Optional[Expression] = None
    then_branch: Optional[Expression] = None
    else_branch: Optional[Expression] = None

@dataclass
class MatchExpr(Expression):
    subject: Optional[Expression] = None
    arms: List['MatchArm'] = field(default_factory=list)

@dataclass
class MatchArm(ASTNode):
    pattern: Optional['Pattern'] = None
    body: Optional[Expression] = None

@dataclass
class GradCall(Expression):
    """Built-in grad() for automatic differentiation"""
    func: Optional[Expression] = None
    wrt: Optional[Expression] = None


# ── Patterns ──────────────────────────────────────────────────────────

@dataclass
class Pattern(ASTNode):
    pass

@dataclass
class WildcardPattern(Pattern):
    pass

@dataclass
class IdentifierPattern(Pattern):
    name: str = ""

@dataclass
class LiteralPattern(Pattern):
    value: Any = None

@dataclass
class ConstructorPattern(Pattern):
    name: str = ""
    args: List[Pattern] = field(default_factory=list)

@dataclass
class ListPattern(Pattern):
    elements: List[Pattern] = field(default_factory=list)


# ── Statements ────────────────────────────────────────────────────────

@dataclass
class Statement(ASTNode):
    pass

@dataclass
class LetStatement(Statement):
    name: str = ""
    type_annotation: Optional[TypeAnnotation] = None
    value: Optional[Expression] = None
    mutable: bool = False

@dataclass
class FunctionDef(Statement):
    name: str = ""
    params: List['Param'] = field(default_factory=list)
    return_type: Optional[TypeAnnotation] = None
    body: List[Statement] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)

@dataclass
class Param(ASTNode):
    name: str = ""
    type_annotation: Optional[TypeAnnotation] = None
    default: Optional[Expression] = None

@dataclass
class ReturnStatement(Statement):
    value: Optional[Expression] = None

@dataclass
class IfStatement(Statement):
    condition: Optional[Expression] = None
    then_body: List[Statement] = field(default_factory=list)
    else_body: List[Statement] = field(default_factory=list)

@dataclass
class ForLoop(Statement):
    var: str = ""
    iterable: Optional[Expression] = None
    body: List[Statement] = field(default_factory=list)

@dataclass
class WhileLoop(Statement):
    condition: Optional[Expression] = None
    body: List[Statement] = field(default_factory=list)

@dataclass
class ExprStatement(Statement):
    expr: Optional[Expression] = None

@dataclass
class ImportStatement(Statement):
    module: str = ""
    names: List[str] = field(default_factory=list)  # empty = import all
    alias: Optional[str] = None

@dataclass
class ModelDef(Statement):
    name: str = ""
    fields: dict = field(default_factory=dict)

@dataclass
class SchemaDef(Statement):
    name: str = ""
    fields: List['SchemaField'] = field(default_factory=list)

@dataclass
class SchemaField(ASTNode):
    name: str = ""
    type_annotation: Optional[TypeAnnotation] = None
    constraint: Optional[Expression] = None

@dataclass
class TypeAlias(Statement):
    name: str = ""
    definition: Optional[TypeAnnotation] = None

@dataclass
class Decorator(ASTNode):
    name: str = ""
    args: List[Expression] = field(default_factory=list)


# ── Program ───────────────────────────────────────────────────────────

@dataclass
class Program(ASTNode):
    statements: List[Statement] = field(default_factory=list)
