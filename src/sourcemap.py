"""
Synaphe Source Map System v0.4.0

Solves the "Black Box" criticism: when transpiled Python crashes,
the traceback now shows the original Synaphe source, not the
generated Python code.

Like TypeScript → JavaScript source maps, this maps every generated
Python line back to the Synaphe source that produced it.
"""

import sys
import traceback
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SourceMapping:
    """Maps a generated Python line to the original Synaphe source."""
    python_line: int
    synaphe_line: int
    synaphe_col: int
    synaphe_source: str  # The original Synaphe line
    synaphe_file: str = "<repl>"


@dataclass
class SourceMap:
    """Complete source map for a transpiled file."""
    synaphe_file: str
    synaphe_source: str  # Full original source
    mappings: Dict[int, SourceMapping] = field(default_factory=dict)

    def add(self, python_line: int, synaphe_line: int, synaphe_col: int = 0,
            synaphe_source: str = ""):
        self.mappings[python_line] = SourceMapping(
            python_line=python_line,
            synaphe_line=synaphe_line,
            synaphe_col=synaphe_col,
            synaphe_source=synaphe_source,
            synaphe_file=self.synaphe_file
        )

    def lookup(self, python_line: int) -> Optional[SourceMapping]:
        """Find the Synaphe source for a given Python line number."""
        if python_line in self.mappings:
            return self.mappings[python_line]
        # Find nearest preceding mapping
        closest = None
        for pline, mapping in self.mappings.items():
            if pline <= python_line:
                if closest is None or pline > closest:
                    closest = pline
        if closest is not None:
            return self.mappings[closest]
        return None


# Global source map registry
_source_maps: Dict[str, SourceMap] = {}


def register_source_map(filename: str, source_map: SourceMap):
    """Register a source map for a transpiled file."""
    _source_maps[filename] = source_map


def get_source_map(filename: str) -> Optional[SourceMap]:
    """Retrieve the source map for a file."""
    return _source_maps.get(filename)


# ═══════════════════════════════════════════════════════════════════
# SOURCE-MAPPED TRANSPILER OUTPUT
# ═══════════════════════════════════════════════════════════════════

def emit_with_source_map(python_code: str, synaphe_source: str,
                          filename: str = "<repl>") -> Tuple[str, SourceMap]:
    """
    Add source map comments to transpiled Python code.

    Input:  plain transpiled Python
    Output: Python with # synaphe:L:C comments + a SourceMap object
    """
    source_map = SourceMap(synaphe_file=filename, synaphe_source=synaphe_source)
    synaphe_lines = synaphe_source.split('\n')

    output_lines = []
    py_line_num = 0

    for line in python_code.split('\n'):
        py_line_num += 1

        # Try to find the corresponding Synaphe line
        # Heuristic: match variable names, function names, keywords
        matched_syn_line = _find_matching_source(line, synaphe_lines)

        if matched_syn_line is not None:
            syn_text = synaphe_lines[matched_syn_line].strip()
            source_map.add(py_line_num, matched_syn_line + 1, 0, syn_text)
            output_lines.append(f"{line}  # synaphe:L{matched_syn_line + 1} {syn_text}")
        else:
            output_lines.append(line)

    return '\n'.join(output_lines), source_map


def _find_matching_source(python_line: str, synaphe_lines: List[str]) -> Optional[int]:
    """
    Heuristic matching: find which Synaphe line produced this Python line.
    Matches on variable names, function names, and structural patterns.
    """
    py_stripped = python_line.strip()
    if not py_stripped or py_stripped.startswith('#') or py_stripped.startswith('"""'):
        return None
    if py_stripped.startswith('import ') or py_stripped.startswith('from '):
        # Try to match import statements
        for i, syn in enumerate(synaphe_lines):
            if syn.strip().startswith('import ') or syn.strip().startswith('from '):
                if any(word in py_stripped for word in syn.strip().split()[-2:]):
                    return i
        return None

    # Extract identifiers from the Python line
    py_identifiers = set(re.findall(r'\b[a-zA-Z_]\w*\b', py_stripped))
    # Remove Python-only keywords
    py_only = {'self', 'def', 'class', 'super', 'True', 'False', 'None',
               'nn', 'torch', 'import', 'from', 'print', 'return',
               'and', 'or', 'not', 'pass', 'lambda'}
    py_identifiers -= py_only

    best_match = None
    best_score = 0

    for i, syn_line in enumerate(synaphe_lines):
        syn_stripped = syn_line.strip()
        if not syn_stripped or syn_stripped.startswith('//'):
            continue

        syn_identifiers = set(re.findall(r'\b[a-zA-Z_]\w*\b', syn_stripped))
        # Remove Synaphe-only keywords
        syn_only = {'let', 'fn', 'model', 'schema', 'where'}
        syn_identifiers -= syn_only

        overlap = len(py_identifiers & syn_identifiers)
        if overlap > best_score:
            best_score = overlap
            best_match = i

    if best_score >= 1:
        return best_match
    return None


# ═══════════════════════════════════════════════════════════════════
# EXCEPTION HOOK — REWRITES TRACEBACKS
# ═══════════════════════════════════════════════════════════════════

def _synaphe_exception_hook(exc_type, exc_value, exc_tb):
    """
    Custom exception hook that rewrites Python tracebacks
    to show Synaphe source locations.

    Instead of:
        File "generated.py", line 47, in _synaphe_pipeline
            result = fn(result)

    You see:
        File "my_program.synaphe", line 3
            data |> normalize |> model.forward
        → Pipeline stage 'normalize' raised: ValueError: ...
    """
    # Get the standard traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)

    # Try to rewrite with source maps
    rewritten = []
    for line in tb_lines:
        # Look for "File "...", line N" patterns
        match = re.match(r'  File "([^"]+)", line (\d+)', line)
        if match:
            filename = match.group(1)
            line_num = int(match.group(2))

            source_map = get_source_map(filename)
            if source_map:
                mapping = source_map.lookup(line_num)
                if mapping:
                    rewritten.append(
                        f'  File "{mapping.synaphe_file}", line {mapping.synaphe_line}\n'
                    )
                    rewritten.append(
                        f'    {mapping.synaphe_source}\n'
                    )
                    continue

        rewritten.append(line)

    # Print rewritten traceback
    sys.stderr.write(''.join(rewritten))


def install_source_map_hook():
    """Install the Synaphe exception hook for source-mapped tracebacks."""
    sys.excepthook = _synaphe_exception_hook


def uninstall_source_map_hook():
    """Restore the default Python exception hook."""
    sys.excepthook = sys.__excepthook__


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════

def format_synaphe_error(error: Exception, source_map: SourceMap = None,
                          python_line: int = None) -> str:
    """Format an error message with Synaphe source context."""
    parts = [f"Synaphe Error: {type(error).__name__}: {error}"]

    if source_map and python_line:
        mapping = source_map.lookup(python_line)
        if mapping:
            parts.append(f"  at {mapping.synaphe_file}:{mapping.synaphe_line}")
            parts.append(f"  → {mapping.synaphe_source}")

            # Show surrounding context
            lines = source_map.synaphe_source.split('\n')
            start = max(0, mapping.synaphe_line - 2)
            end = min(len(lines), mapping.synaphe_line + 2)
            parts.append("")
            for i in range(start, end):
                marker = "→ " if i == mapping.synaphe_line - 1 else "  "
                parts.append(f"  {marker}{i+1:3d} | {lines[i]}")

    return '\n'.join(parts)


__all__ = [
    'SourceMap', 'SourceMapping',
    'emit_with_source_map', 'register_source_map', 'get_source_map',
    'install_source_map_hook', 'uninstall_source_map_hook',
    'format_synaphe_error',
]
