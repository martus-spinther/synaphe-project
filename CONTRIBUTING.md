# Contributing to Synaphe

Thank you for your interest in contributing to Synaphe! Whether you're fixing a typo, adding a test, implementing a new quantum gate, or proposing a language feature — every contribution matters.

## Code of Conduct

By participating, you agree to maintain a welcoming, respectful environment. Be kind. Be constructive. Assume good intentions.

## How to Contribute

### Reporting Bugs

Open a GitHub Issue with:
- A clear title describing the problem
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Your Python version and OS

### Suggesting Features

Open a GitHub Issue with the `feature request` label. Include:
- The problem you're trying to solve
- Your proposed solution (syntax examples help!)
- How it fits with Synaphe's three pillars (shape safety, linear types, autodiff)

### Your First Contribution

Look for issues labeled **`good first issue`** — these are scoped, approachable tasks with clear descriptions. Some examples:

- Add a new quantum gate to `stdlib/runtime.py`
- Add a new hardware profile to `src/hardware.py`
- Write a test for an edge case
- Improve an error message
- Add a code example

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/martus-spinther/synaphe.git
cd synaphe

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install numpy  # Required
pip install torch pennylane qiskit  # Optional, for full backend support

# Run the test suite
python tests/test_parser.py       # 47 tests
python tests/test_typechecker.py  # 19 tests
python tests/test_v030.py         # 20 tests

# Run the demo
python examples/demo.py
```

### Pull Request Process

1. **Fork** the repository and create a branch from `main`
2. **Write tests** for any new functionality
3. **Ensure all 86 tests pass** before submitting
4. **Update documentation** if you've changed APIs or added features
5. **Open a PR** with a clear description of what and why

### Code Style

- Python 3.9+ compatible
- Type hints encouraged
- Docstrings for all public functions
- Descriptive variable names (no single letters except loop counters and math)
- Error messages should be helpful — tell the user what went wrong AND how to fix it

### Project Structure

| Directory | Purpose | Good for |
|-----------|---------|----------|
| `src/lexer.py` | Tokenizer | Adding new operators/keywords |
| `src/parser.py` | Parser | New syntax constructs |
| `src/types.py` | Type definitions | New types (e.g., `StreamType`) |
| `src/typechecker.py` | Type checking | Better error messages, new checks |
| `src/hardware.py` | Hardware profiles | Adding new QPU specifications |
| `src/transpiler.py` | Python codegen | New transpilation targets |
| `stdlib/runtime.py` | Quantum runtime | New gates, algorithms, optimizers |
| `stdlib/data.py` | Data pipeline | New data formats, transforms |
| `tests/` | Test suite | Edge cases, regression tests |
| `examples/` | Examples | Tutorials, use cases |

### Areas Where Help Is Especially Welcome

**Standard Library**
- Quantum algorithms: Grover's search, quantum phase estimation, QAOA variants
- Error mitigation: zero-noise extrapolation, probabilistic error cancellation
- Classical ML: training loops, evaluation metrics, model serialization

**Hardware Profiles**
- Quantinuum (trapped ion)
- Amazon Braket backends
- Custom simulator configurations

**Type System**
- Better type inference (reduce the need for explicit annotations)
- Dependent types for circuit depth tracking
- Effect system for side-effect tracking in quantum operations

**Developer Experience**
- Language server protocol (LSP) implementation
- Syntax highlighting for VS Code, Vim, Emacs
- Jupyter kernel for notebook integration
- Better REPL with tab completion and history

**Documentation**
- Tutorials for specific use cases (drug discovery, portfolio optimization)
- Video walkthroughs
- Translations to other languages

## Questions?

Open a GitHub Discussion or file an issue. There are no bad questions — quantum computing is hard, and we're all learning together.
