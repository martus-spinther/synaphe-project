"""
Synaphe Type Checker v0.3.0

The "key" to the type system's "lock."
Walks the AST and enforces three invariants:

1. TENSOR SHAPE SAFETY: Tensor<f32, [64, 10]> piped into a layer
   expecting [32, 10] is a compile error, not a RuntimeError.

2. LINEAR QUANTUM SAFETY: QState variables must be used exactly once.
   Reusing a qubit after measurement, or cloning a quantum state,
   is a compile error — enforcing the No-Cloning Theorem.

3. DIFFERENTIABLE TRACKING: @differentiable functions have their
   parameters tracked so grad() can apply the Parameter Shift Rule
   for quantum gates or autograd for classical tensors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto

from .types import (
    NovaType, TypeKind, IntType, FloatType, BoolType, StringType, NoneType,
    TensorType, DimLit, DimVar, Dim,
    QubitType, QRegisterType, QStateType, MeasurementType, QubitState,
    GradType, DiffParam,
    FunctionSig, TupleType, ListType,
    ShapeError, matmul_shape, broadcast_shape,
    HardwareTarget
)
# Import AST nodes — rename TensorType to avoid collision
from .ast_nodes import (
    ASTNode, Program, Statement, Expression,
    LetStatement, FunctionDef, ReturnStatement, IfStatement,
    ForLoop, WhileLoop, ExprStatement, ImportStatement,
    ModelDef, SchemaDef, TypeAlias, Param,
    IntLiteral, FloatLiteral, StringLiteral, BoolLiteral,
    Identifier, BinaryOp, UnaryOp, FunctionCall, MethodCall,
    MemberAccess, IndexAccess, Pipeline, MatMul, ListLiteral,
    DictLiteral, MatchExpr, MatchArm, Lambda, IfExpr, GradCall,
    TypeAnnotation, SimpleType, GenericType, FunctionType,
)
from .ast_nodes import TensorType as ASTTensorType


# ── Error Types ──────────────────────────────────────────────────────

class TypeCheckError(Exception):
    """Base class for type check errors."""
    pass


class ShapeMismatchError(TypeCheckError):
    """Tensor shapes don't match."""
    pass


class LinearityError(TypeCheckError):
    """Quantum resource used incorrectly (copied, reused after measure, leaked)."""
    pass


class GradientError(TypeCheckError):
    """Differentiation error (non-differentiable op in differentiable context)."""
    pass


# ── Quantum Resource Tracker ─────────────────────────────────────────

@dataclass
class QuantumResource:
    """Tracks the lifecycle of a single quantum resource."""
    name: str
    qtype: NovaType
    state: QubitState = QubitState.FRESH
    entangled_with: Set[str] = field(default_factory=set)
    defined_at: Tuple[int, int] = (0, 0)  # line, col
    used_at: List[Tuple[int, int]] = field(default_factory=list)

    def mark_active(self, line: int, col: int):
        if self.state == QubitState.MEASURED:
            raise LinearityError(
                f"Quantum resource '{self.name}' used after measurement at L{line}:{col}. "
                f"Once measured, a quantum state collapses to classical — it cannot be "
                f"used as a quantum input again. (No-Cloning Theorem)"
            )
        if self.state == QubitState.CONSUMED:
            raise LinearityError(
                f"Quantum resource '{self.name}' already consumed at L{line}:{col}. "
                f"Quantum states are LINEAR types — they can only be used once. "
                f"If you need the value again, re-prepare the state."
            )
        self.state = QubitState.ACTIVE
        self.used_at.append((line, col))

    def mark_measured(self, line: int, col: int):
        self.state = QubitState.MEASURED
        self.used_at.append((line, col))

    def mark_consumed(self, line: int, col: int):
        self.state = QubitState.CONSUMED
        self.used_at.append((line, col))


# ── Type Environment ─────────────────────────────────────────────────

@dataclass
class TypeEnv:
    """
    The type environment tracks:
    - Variable types (name -> NovaType)
    - Quantum resource states (name -> QuantumResource)
    - Function signatures (name -> FunctionSig)
    - Whether we're inside a @differentiable context
    """
    variables: Dict[str, NovaType] = field(default_factory=dict)
    quantum_resources: Dict[str, QuantumResource] = field(default_factory=dict)
    functions: Dict[str, FunctionSig] = field(default_factory=dict)
    in_differentiable: bool = False
    diff_params: Set[str] = field(default_factory=set)
    errors: List[TypeCheckError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def child_scope(self) -> 'TypeEnv':
        """Create a child scope that inherits parent bindings."""
        child = TypeEnv(
            variables=dict(self.variables),
            quantum_resources=dict(self.quantum_resources),
            functions=dict(self.functions),
            in_differentiable=self.in_differentiable,
            diff_params=set(self.diff_params),
            errors=self.errors,  # shared error list
            warnings=self.warnings,
        )
        return child

    def bind(self, name: str, typ: NovaType, line: int = 0, col: int = 0):
        """Bind a variable to a type."""
        # Check for quantum resource rebinding
        if name in self.quantum_resources:
            res = self.quantum_resources[name]
            if res.state == QubitState.ACTIVE:
                self.warnings.append(
                    f"Warning L{line}: Rebinding quantum resource '{name}' — "
                    f"previous quantum state will be leaked. Consider measuring first."
                )

        self.variables[name] = typ

        # Track quantum resources
        if typ.is_quantum():
            self.quantum_resources[name] = QuantumResource(
                name=name, qtype=typ, state=QubitState.FRESH,
                defined_at=(line, col)
            )

    def lookup(self, name: str) -> Optional[NovaType]:
        return self.variables.get(name)

    def use_quantum(self, name: str, line: int, col: int):
        """Mark a quantum resource as used (linear consumption)."""
        if name in self.quantum_resources:
            self.quantum_resources[name].mark_active(line, col)

    def measure_quantum(self, name: str, line: int, col: int):
        """Mark a quantum resource as measured (collapse to classical)."""
        if name in self.quantum_resources:
            self.quantum_resources[name].mark_measured(line, col)

    def check_quantum_leaks(self):
        """At scope exit, check that all quantum resources were used."""
        for name, res in self.quantum_resources.items():
            if res.state == QubitState.FRESH:
                self.warnings.append(
                    f"Warning: Quantum resource '{name}' allocated but never used "
                    f"(defined at L{res.defined_at[0]}). This wastes qubits."
                )


# ── Standard Library Type Signatures ─────────────────────────────────

def build_stdlib_env() -> TypeEnv:
    """
    Pre-populate the type environment with standard library signatures.
    This is the HardwareMap — each function knows what it maps to.
    """
    env = TypeEnv()

    f32 = FloatType(bits=32)
    f64 = FloatType(bits=64)
    int_t = IntType()
    bool_t = BoolType()
    str_t = StringType()

    # ── Tensor operations ────────────────────────────────────────
    env.functions["randn"] = FunctionSig(
        params=[("shape", int_t)],  # variadic in practice
        return_type=TensorType(dtype="Float32")
    )
    env.functions["zeros"] = FunctionSig(
        params=[("shape", int_t)],
        return_type=TensorType(dtype="Float32")
    )
    env.functions["ones"] = FunctionSig(
        params=[("shape", int_t)],
        return_type=TensorType(dtype="Float32")
    )
    env.functions["reshape"] = FunctionSig(
        params=[("tensor", TensorType()), ("shape", int_t)],
        return_type=TensorType(dtype="Float32")
    )
    env.functions["softmax"] = FunctionSig(
        params=[("input", TensorType())],
        return_type=TensorType(dtype="Float32")
    )
    env.functions["relu"] = FunctionSig(
        params=[("input", TensorType())],
        return_type=TensorType(dtype="Float32")
    )
    env.functions["tanh"] = FunctionSig(
        params=[("input", TensorType())],
        return_type=TensorType(dtype="Float32")
    )
    env.functions["sigmoid"] = FunctionSig(
        params=[("input", TensorType())],
        return_type=TensorType(dtype="Float32")
    )
    env.functions["concat"] = FunctionSig(
        params=[("tensors", ListType(element_type=TensorType()))],
        return_type=TensorType(dtype="Float32")
    )
    env.functions["dot"] = FunctionSig(
        params=[("a", TensorType()), ("b", TensorType())],
        return_type=f32
    )

    # ── Quantum operations ───────────────────────────────────────
    # These have LINEAR parameter types — the type checker enforces
    # that quantum arguments are consumed, not copied.

    env.functions["qubit"] = FunctionSig(
        params=[],
        return_type=QubitType()
    )
    env.functions["qregister"] = FunctionSig(
        params=[("n", int_t)],
        return_type=QRegisterType()  # n_qubits filled at call site
    )

    # Quantum gates: take LINEAR QState, return new LINEAR QState
    # The old state is CONSUMED (cannot be reused)
    for gate in ["hadamard", "rx", "ry", "rz", "cx", "cz", "swap",
                 "prepare_hartree_fock", "single_excitation",
                 "double_excitation", "entangle_layer"]:
        env.functions[gate] = FunctionSig(
            params=[("state", QStateType(kind=TypeKind.LINEAR))],
            return_type=QStateType(kind=TypeKind.LINEAR)
        )

    # Measurement: LINEAR QState -> CLASSICAL Measurement
    # This is the quantum-classical boundary!
    env.functions["measure"] = FunctionSig(
        params=[("state", QStateType(kind=TypeKind.LINEAR))],
        return_type=MeasurementType()  # CLASSICAL — can be freely copied
    )
    env.functions["measure_all"] = FunctionSig(
        params=[],
        return_type=MeasurementType()
    )

    # Expectation: QState + Operator -> Float (CLASSICAL)
    env.functions["expectation"] = FunctionSig(
        params=[("operator", str_t), ("state", QStateType())],
        return_type=f32
    )

    # ── Hybrid operations ────────────────────────────────────────
    env.functions["hamiltonian"] = FunctionSig(
        params=[("molecule", str_t)],
        return_type=TensorType(dtype="Complex64")  # Operator matrix
    )
    env.functions["minimize"] = FunctionSig(
        params=[("cost_fn", FunctionSig(return_type=f32))],
        return_type=TupleType(elements=(f32, TensorType()))
    )
    env.functions["qaoa"] = FunctionSig(
        params=[("cost", FunctionSig(return_type=f32)), ("qubits", int_t)],
        return_type=TupleType(elements=(f32, QStateType()))
    )

    # ── Differentiation ──────────────────────────────────────────
    # grad() is the bridge: classical -> autograd, quantum -> parameter shift
    env.functions["grad"] = FunctionSig(
        params=[("fn", FunctionSig(return_type=f32))],
        return_type=GradType(inner_type=TensorType())
    )

    # ── IO ────────────────────────────────────────────────────────
    env.functions["print"] = FunctionSig(
        params=[("value", str_t)],
        return_type=NoneType()
    )
    env.functions["load"] = FunctionSig(
        params=[("path", str_t)],
        return_type=TensorType()
    )

    # ── Layers (for model definitions) ───────────────────────────
    env.functions["Linear"] = FunctionSig(
        params=[("in_features", int_t), ("out_features", int_t)],
        return_type=FunctionSig(
            params=[("input", TensorType())],
            return_type=TensorType()
        )
    )
    env.functions["Dropout"] = FunctionSig(
        params=[("p", f32)],
        return_type=FunctionSig(
            params=[("input", TensorType())],
            return_type=TensorType()
        )
    )

    return env


# ── The Type Checker ─────────────────────────────────────────────────

class TypeChecker:
    """
    Walks the AST and enforces all three pillars:
    1. Tensor shape safety
    2. Linear quantum safety
    3. Differentiable tracking
    """

    def __init__(self):
        self.env = build_stdlib_env()
        self.errors: List[TypeCheckError] = []
        self.warnings: List[str] = []

    def check(self, program: Program) -> Tuple[List[TypeCheckError], List[str]]:
        """Type-check an entire program. Returns (errors, warnings)."""
        self.env.errors = self.errors
        self.env.warnings = self.warnings

        for stmt in program.statements:
            self.check_statement(stmt, self.env)

        # Check for leaked quantum resources
        self.env.check_quantum_leaks()

        return self.errors, self.warnings

    def check_statement(self, stmt: Statement, env: TypeEnv):
        """Type-check a statement and update the environment."""
        try:
            if isinstance(stmt, LetStatement):
                self._check_let(stmt, env)
            elif isinstance(stmt, FunctionDef):
                self._check_function(stmt, env)
            elif isinstance(stmt, IfStatement):
                self._check_if(stmt, env)
            elif isinstance(stmt, ForLoop):
                self._check_for(stmt, env)
            elif isinstance(stmt, ReturnStatement):
                if stmt.value:
                    self.infer_type(stmt.value, env)
            elif isinstance(stmt, ExprStatement):
                self.infer_type(stmt.expr, env)
            elif isinstance(stmt, ModelDef):
                self._check_model(stmt, env)
            elif isinstance(stmt, SchemaDef):
                self._check_schema(stmt, env)
            elif isinstance(stmt, ImportStatement):
                pass  # Imports don't need type checking
        except TypeCheckError as e:
            self.errors.append(e)

    def _check_let(self, stmt: LetStatement, env: TypeEnv):
        """Type-check a let binding."""
        inferred = self.infer_type(stmt.value, env)
        declared = None
        
        if stmt.type_annotation:
            declared = self._resolve_annotation(stmt.type_annotation)
            if declared and inferred:
                self._check_compatible(declared, inferred, stmt.line, stmt.col)

        # Use declared type if it's more specific (has shape info)
        if declared and isinstance(declared, TensorType) and declared.shape:
            result_type = declared
        elif inferred:
            result_type = inferred
        else:
            result_type = NoneType()
            
        env.bind(stmt.name, result_type, stmt.line, stmt.col)

    def _check_function(self, stmt: FunctionDef, env: TypeEnv):
        """Type-check a function definition."""
        is_diff = any("differentiable" in d for d in stmt.decorators)

        # Build function signature
        params = []
        fn_env = env.child_scope()
        fn_env.in_differentiable = is_diff

        for p in stmt.params:
            p_type = self._resolve_annotation(p.type_annotation) if p.type_annotation else NoneType()
            params.append((p.name, p_type))
            fn_env.bind(p.name, p_type, p.line, p.col)

            # Track differentiable params
            if is_diff and isinstance(p_type, (FloatType, TensorType)):
                fn_env.diff_params.add(p.name)

        ret_type = self._resolve_annotation(stmt.return_type) if stmt.return_type else NoneType()

        sig = FunctionSig(params=params, return_type=ret_type)
        env.functions[stmt.name] = sig
        env.bind(stmt.name, sig, stmt.line, stmt.col)

        # Check body
        for s in stmt.body:
            self.check_statement(s, fn_env)

        # At function exit, verify linear resources
        fn_env.check_quantum_leaks()

    def _check_if(self, stmt: IfStatement, env: TypeEnv):
        cond_type = self.infer_type(stmt.condition, env)
        then_env = env.child_scope()
        for s in stmt.then_body:
            self.check_statement(s, then_env)
        if stmt.else_body:
            else_env = env.child_scope()
            for s in stmt.else_body:
                self.check_statement(s, else_env)

    def _check_for(self, stmt: ForLoop, env: TypeEnv):
        loop_env = env.child_scope()
        iter_type = self.infer_type(stmt.iterable, env)
        loop_env.bind(stmt.var, IntType(), stmt.line, stmt.col)
        for s in stmt.body:
            self.check_statement(s, loop_env)

    def _check_model(self, stmt: ModelDef, env: TypeEnv):
        """Type-check a model definition and register it."""
        env.bind(stmt.name, TensorType(), stmt.line, stmt.col)

    def _check_schema(self, stmt: SchemaDef, env: TypeEnv):
        """Register a schema as a type."""
        env.bind(stmt.name, TensorType(), stmt.line, stmt.col)

    # ── Type Inference ───────────────────────────────────────────────

    def infer_type(self, expr: Expression, env: TypeEnv) -> Optional[NovaType]:
        """Infer the type of an expression."""
        if expr is None:
            return NoneType()

        if isinstance(expr, IntLiteral):
            return IntType()
        if isinstance(expr, FloatLiteral):
            return FloatType(bits=32)
        if isinstance(expr, StringLiteral):
            return StringType()
        if isinstance(expr, BoolLiteral):
            return BoolType()

        if isinstance(expr, Identifier):
            return self._infer_identifier(expr, env)

        if isinstance(expr, BinaryOp):
            return self._infer_binary(expr, env)

        if isinstance(expr, MatMul):
            return self._infer_matmul(expr, env)

        if isinstance(expr, Pipeline):
            return self._infer_pipeline(expr, env)

        if isinstance(expr, FunctionCall):
            return self._infer_call(expr, env)

        if isinstance(expr, MethodCall):
            return self._infer_method_call(expr, env)

        if isinstance(expr, MemberAccess):
            obj_type = self.infer_type(expr.object, env)
            return NoneType()  # Simplified for now

        if isinstance(expr, IndexAccess):
            return self._infer_index(expr, env)

        if isinstance(expr, ListLiteral):
            if expr.elements:
                elem_type = self.infer_type(expr.elements[0], env)
                return ListType(element_type=elem_type)
            return ListType(element_type=NoneType())

        if isinstance(expr, UnaryOp):
            return self.infer_type(expr.operand, env)

        if isinstance(expr, MatchExpr):
            return NoneType()  # Simplified

        return NoneType()

    def _infer_identifier(self, expr: Identifier, env: TypeEnv) -> Optional[NovaType]:
        typ = env.lookup(expr.name)
        if typ is None:
            # Check functions
            if expr.name in env.functions:
                return env.functions[expr.name]
            return NoneType()

        # LINEAR CHECK: if this is a quantum resource, track usage
        if typ.is_quantum() and expr.name in env.quantum_resources:
            try:
                env.use_quantum(expr.name, expr.line, expr.col)
            except LinearityError as e:
                self.errors.append(e)

        return typ

    def _infer_binary(self, expr: BinaryOp, env: TypeEnv) -> Optional[NovaType]:
        left = self.infer_type(expr.left, env)
        right = self.infer_type(expr.right, env)

        # Tensor shape checking for arithmetic
        if isinstance(left, TensorType) and isinstance(right, TensorType):
            if expr.op in ('+', '-', '*', '/'):
                try:
                    return broadcast_shape(left, right)
                except ShapeError as e:
                    self.errors.append(ShapeMismatchError(str(e)))
                    return left

        # Comparison ops return Bool
        if expr.op in ('==', '!=', '<', '>', '<=', '>='):
            return BoolType()

        # Boolean ops return Bool
        if expr.op in ('&&', '||'):
            return BoolType()

        # Numeric arithmetic
        if isinstance(left, FloatType) or isinstance(right, FloatType):
            return FloatType(bits=32)
        if isinstance(left, IntType) and isinstance(right, IntType):
            return IntType()

        return left or right or NoneType()

    def _infer_matmul(self, expr: MatMul, env: TypeEnv) -> Optional[NovaType]:
        """
        PILLAR 1: Tensor shape checking for matrix multiplication.
        This is where we catch the #1 PyTorch runtime error at compile time.
        """
        left = self.infer_type(expr.left, env)
        right = self.infer_type(expr.right, env)

        if isinstance(left, TensorType) and isinstance(right, TensorType):
            try:
                return matmul_shape(left, right)
            except ShapeError as e:
                self.errors.append(ShapeMismatchError(
                    f"L{expr.line}:{expr.col}: {e}"
                ))
                return TensorType()

        return TensorType()

    def _infer_pipeline(self, expr: Pipeline, env: TypeEnv) -> Optional[NovaType]:
        """
        Type-check a pipeline chain: expr |> fn1 |> fn2 |> fn3
        The output type of each stage becomes the input to the next.
        """
        current_type = self.infer_type(expr.stages[0], env)

        for stage in expr.stages[1:]:
            stage_type = self.infer_type(stage, env)

            # If the stage is a function, verify input compatibility
            if isinstance(stage_type, FunctionSig):
                if stage_type.params:
                    expected = stage_type.params[0][1]
                    self._check_pipeline_compat(current_type, expected,
                                                stage.line if hasattr(stage, 'line') else 0)
                current_type = stage_type.return_type

                # QUANTUM LINEAR CHECK: if piping a quantum state into measure,
                # the state is consumed and result is classical
                if isinstance(current_type, MeasurementType):
                    # Mark the quantum source as measured
                    if isinstance(expr.stages[0], Identifier):
                        name = expr.stages[0].name
                        if name in env.quantum_resources:
                            env.measure_quantum(name, expr.line, expr.col)

            elif isinstance(stage_type, TensorType):
                current_type = stage_type
            # Otherwise, assume the stage transforms the type
            else:
                pass  # Keep current type

        return current_type

    def _infer_call(self, expr: FunctionCall, env: TypeEnv) -> Optional[NovaType]:
        """Type-check a function call."""
        callee_name = ""
        if isinstance(expr.callee, Identifier):
            callee_name = expr.callee.name

        # Special handling for grad()
        if callee_name == "grad":
            return self._check_grad_call(expr, env)

        # Special handling for qregister(n) — create sized type
        if callee_name == "qregister" and expr.args:
            n_arg = expr.args[0]
            if isinstance(n_arg, IntLiteral):
                result = QRegisterType(n_qubits=n_arg.value)
                return result

        # Look up function signature
        if callee_name in env.functions:
            sig = env.functions[callee_name]

            # Check argument types against parameters
            for i, arg in enumerate(expr.args):
                arg_type = self.infer_type(arg, env)
                if i < len(sig.params):
                    param_name, param_type = sig.params[i]

                    # LINEAR CHECK: quantum arguments are consumed
                    if param_type.is_quantum() and isinstance(arg, Identifier):
                        if arg.name in env.quantum_resources:
                            try:
                                env.quantum_resources[arg.name].mark_consumed(
                                    expr.line, expr.col
                                )
                            except LinearityError as e:
                                self.errors.append(e)

            return sig.return_type

        # Infer from callee expression
        callee_type = self.infer_type(expr.callee, env)
        if isinstance(callee_type, FunctionSig):
            return callee_type.return_type

        return NoneType()

    def _check_grad_call(self, expr: FunctionCall, env: TypeEnv) -> Optional[NovaType]:
        """
        PILLAR 3: Type-check a grad() call.

        grad(fn) checks:
        1. fn must be @differentiable or contain only differentiable ops
        2. For quantum functions: auto-applies Parameter Shift Rule
        3. For classical functions: auto-applies backpropagation
        4. Returns Grad<T> where T matches fn's return type
        """
        if not expr.args:
            self.errors.append(GradientError(
                f"L{expr.line}: grad() requires a function argument"
            ))
            return GradType(inner_type=FloatType())

        fn_type = self.infer_type(expr.args[0], env)

        if isinstance(fn_type, FunctionSig):
            # Check that the function returns a scalar (required for grad)
            ret = fn_type.return_type
            if isinstance(ret, TensorType) and ret.ndim() > 0:
                self.warnings.append(
                    f"L{expr.line}: grad() on a function returning a tensor will "
                    f"compute the Jacobian. For scalar loss, this is the gradient."
                )

            # Check for quantum parameters — will use Parameter Shift Rule
            has_quantum_params = any(
                p[1].is_quantum() for p in fn_type.params
            )
            if has_quantum_params:
                self.warnings.append(
                    f"L{expr.line}: grad() on quantum function — using Parameter Shift Rule. "
                    f"This requires 2 circuit evaluations per parameter."
                )

            return GradType(inner_type=fn_type.return_type)

        return GradType(inner_type=FloatType())

    def _infer_method_call(self, expr: MethodCall, env: TypeEnv) -> Optional[NovaType]:
        obj_type = self.infer_type(expr.object, env)
        # Common tensor methods
        if isinstance(obj_type, TensorType):
            if expr.method == "argmax":
                return IntType()
            if expr.method in ("mean", "sum", "std", "var", "norm"):
                return FloatType()
            if expr.method == "shape":
                return ListType(element_type=IntType())
            if expr.method == "T":
                if obj_type.shape and len(obj_type.shape) == 2:
                    return TensorType(
                        dtype=obj_type.dtype,
                        shape=(obj_type.shape[1], obj_type.shape[0])
                    )
        return NoneType()

    def _infer_index(self, expr: IndexAccess, env: TypeEnv) -> Optional[NovaType]:
        obj_type = self.infer_type(expr.object, env)
        if isinstance(obj_type, TensorType) and obj_type.shape:
            # Indexing removes the first dimension
            return TensorType(dtype=obj_type.dtype, shape=obj_type.shape[1:])
        if isinstance(obj_type, ListType):
            return obj_type.element_type
        return NoneType()

    # ── Compatibility Checks ─────────────────────────────────────────

    def _check_compatible(self, declared: NovaType, inferred: NovaType,
                          line: int, col: int):
        """Check that inferred type is compatible with declared type."""
        # Import here to distinguish AST TensorType from types.TensorType
        from . import ast_nodes as ast
        
        # Both must be NovaType instances for deep checking
        if not hasattr(declared, 'is_quantum') or not hasattr(inferred, 'is_quantum'):
            return  # Can't check non-NovaType objects
            
        if isinstance(declared, TensorType) and isinstance(inferred, TensorType):
            if declared.shape and inferred.shape:
                if len(declared.shape) != len(inferred.shape):
                    self.errors.append(ShapeMismatchError(
                        f"L{line}:{col}: Declared shape {declared} but got {inferred} "
                        f"(different number of dimensions)"
                    ))
                    return
                for d, i in zip(declared.shape, inferred.shape):
                    d_val = d.value if isinstance(d, DimLit) else None
                    i_val = i.value if isinstance(i, DimLit) else None
                    if d_val is not None and i_val is not None:
                        if d_val != i_val:
                            self.errors.append(ShapeMismatchError(
                                f"L{line}:{col}: Declared {declared} but got {inferred}"
                            ))
                            return

        # Check quantum type compatibility
        if declared.is_quantum() != inferred.is_quantum():
            self.errors.append(TypeCheckError(
                f"L{line}:{col}: Cannot assign {inferred} to {declared} — "
                f"mixing classical and quantum types"
            ))

    def _check_pipeline_compat(self, actual: NovaType, expected: NovaType, line: int):
        """Check that a pipeline stage's input matches the previous output."""
        if isinstance(actual, TensorType) and isinstance(expected, TensorType):
            if actual.shape and expected.shape:
                for a, e in zip(actual.shape, expected.shape):
                    if isinstance(a, DimLit) and isinstance(e, DimLit):
                        if a.value != e.value:
                            self.errors.append(ShapeMismatchError(
                                f"L{line}: Pipeline shape mismatch — previous stage "
                                f"outputs {actual} but next stage expects {expected}"
                            ))
                            return

    def _resolve_annotation(self, ann) -> Optional[NovaType]:
        """Convert an AST type annotation to a NovaType."""
        if ann is None:
            return None

        if hasattr(ann, 'name'):
            name = ann.name if hasattr(ann, 'name') else str(ann)
            type_map = {
                'Int': IntType(),
                'Float': FloatType(bits=32),
                'Float32': FloatType(bits=32),
                'Float64': FloatType(bits=64),
                'String': StringType(),
                'Bool': BoolType(),
                'Qubit': QubitType(),
            }
            if name in type_map:
                return type_map[name]

        if hasattr(ann, 'dtype'):  # TensorType AST node
            shape = tuple(
                DimLit(d) if isinstance(d, int) else DimVar(d)
                for d in getattr(ann, 'shape', [])
            )
            return TensorType(dtype=getattr(ann, 'dtype', 'Float32'), shape=shape)

        if hasattr(ann, 'params') and hasattr(ann, 'name'):
            name = ann.name
            if name == 'QRegister':
                return QRegisterType(n_qubits=0)
            if name == 'QState':
                return QStateType(n_qubits=0)
            if name == 'Grad':
                return GradType()

        return NoneType()


# ── Convenience ──────────────────────────────────────────────────────

def typecheck(program: Program) -> Tuple[List[TypeCheckError], List[str]]:
    """Type-check a program. Returns (errors, warnings)."""
    checker = TypeChecker()
    return checker.check(program)
