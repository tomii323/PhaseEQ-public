"""Static UI decision inventory and opt-in, payload-free render observations.

Conditions are recorded as source expressions, never evaluated by diagnostics.
An unobserved call is unknown, not evidence that its UI is hidden.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys


DIAGNOSTICS_ENV = "PHASEEQ_UI_VISIBILITY_DIAGNOSTICS"
INVENTORY_JSON = "docs/development/ui-visibility-inventory.json"
INVENTORY_TABLE = "docs/development/ui-visibility-inventory.md"
POLICY_SOURCES = frozenset({
    "utils/ui_visibility.py", "utils/band_split_capability.py",
    "src/fir_design_common/fir_state.py",
})
UI_COMMANDS = frozenset({
    "button", "download_button", "link_button", "checkbox", "toggle", "radio",
    "selectbox", "multiselect", "segmented_control", "pills", "slider",
    "select_slider", "number_input", "text_input", "text_area", "date_input",
    "time_input", "file_uploader", "camera_input", "audio_input", "data_editor",
    "dataframe", "table", "plotly_chart", "altair_chart", "vega_lite_chart",
    "line_chart", "bar_chart", "scatter_chart", "pyplot", "image", "audio",
    "video", "markdown", "caption", "text", "write", "json", "info", "warning",
    "error", "success", "metric", "header", "subheader", "title", "toast",
    "expander", "popover", "tabs", "container", "columns", "empty", "form",
    "form_submit_button", "dialog", "status", "progress", "spinner", "stop",
})


def _ui_call(node):
    if isinstance(node.func, ast.Attribute) and node.func.attr in UI_COMMANDS:
        return node.func.attr
    if isinstance(node.func, ast.Name) and node.func.id.startswith(
        ("render_", "_render_", "localized_", "_series_chart", "_error_chart")
    ):
        return node.func.id
    if isinstance(node.func, ast.Name) and node.func.id in UI_COMMANDS:
        return node.func.id
    return None


class _InventoryVisitor(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.owner = "<module>"
        self.guards = []
        self.rows = []
        self.occurrences = Counter()

    def visit_FunctionDef(self, node):
        previous = self.owner
        self.owner = node.name if previous == "<module>" else previous + "." + node.name
        self.generic_visit(node)
        self.owner = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node):
        self.visit(node.test)
        expression = ast.unparse(node.test)
        for branch, children in ((True, node.body), (False, node.orelse)):
            self.guards.append({"expression": expression, "branch": branch})
            for child in children:
                self.visit(child)
            self.guards.pop()

    def visit_IfExp(self, node):
        self.visit(node.test)
        expression = ast.unparse(node.test)
        for branch, child in ((True, node.body), (False, node.orelse)):
            self.guards.append({"expression": expression, "branch": branch})
            self.visit(child)
            self.guards.pop()

    def visit_Call(self, node):
        command = _ui_call(node)
        if command:
            options = {kw.arg: ast.unparse(kw.value) for kw in node.keywords
                       if kw.arg in {"disabled", "expanded"}}
            identity = json.dumps([self.path, self.owner, ast.unparse(node.func),
                                   self.guards, options], sort_keys=True)
            ordinal = self.occurrences[identity]
            self.occurrences[identity] += 1
            rule_id = "ui." + hashlib.sha256(f"{identity}:{ordinal}".encode()).hexdigest()[:16]
            self.rows.append({
                "id": rule_id, "source": self.path, "owner": self.owner,
                "line": node.lineno, "end_line": node.end_lineno,
                "call": ast.unparse(node.func), "command": command,
                "guards": list(self.guards), **options,
                "kind": "control_flow" if command == "stop" else "render_call",
            })
        self.generic_visit(node)

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return
        for match in re.finditer(r"(?:display\s*:\s*none|visibility\s*:\s*(?:hidden|collapse))\b", node.value):
            preceding = re.search(r"([^{}]+)\{+[^{}]*$", node.value[:match.start()])
            selector = re.sub(r"<[^>]+>", "", preceding.group(1)).strip() if preceding else "<style fragment>"
            identity = json.dumps([self.path, self.owner, selector, match.group(), self.guards], sort_keys=True)
            ordinal = self.occurrences[identity]
            self.occurrences[identity] += 1
            self.rows.append({
                "id": "ui." + hashlib.sha256(f"{identity}:{ordinal}".encode()).hexdigest()[:16],
                "source": self.path, "owner": self.owner,
                "line": node.lineno, "end_line": node.end_lineno,
                "call": selector, "command": "css_visibility",
                "guards": list(self.guards), "kind": "presentation_rule",
                "declaration": match.group(), "location": "enclosing_string_literal",
            })


def build_visibility_inventory(root: Path) -> dict:
    """Inspect both apps and their Streamlit-bearing source modules."""
    paths = [root / "phaseeq.py", root / "composite_streamlit_app.py"]
    paths += [path for directory in ("src", "ui", "utils")
              for path in (root / directory).rglob("*.py")]
    modules, rules = [], []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(source)
        has_streamlit = any(
            (isinstance(node, ast.Import) and any(alias.name == "streamlit" or alias.name.startswith("streamlit.") for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("streamlit"))
            for node in ast.walk(tree)
        )
        if not (has_streamlit or relative.startswith("ui/") or relative in POLICY_SOURCES
                or path.name in {"phaseeq.py", "composite_streamlit_app.py"}):
            continue
        visitor = _InventoryVisitor(relative)
        visitor.visit(tree)
        rules.extend(visitor.rows)
        modules.append({"source": relative,
                        "kind": "streamlit_source" if has_streamlit else "policy_or_ui_module",
                        "sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "rules": len(visitor.rows)})
    return {"schema_version": 1, "scope": "PhaseEQ and embedded Multiway Streamlit sources",
            "limitations": ["Expressions are not evaluated.",
                             "Not observed is unknown, not hidden.",
                             "Caller guards do not imply a function definition was executed.",
                             "Dynamic receiver methods are static candidates, not proven Streamlit calls.",
                             "Embedded CSS declarations are source rules, not live DOM visibility."],
            "modules": modules, "rules": rules}


def inventory_json(inventory):
    return json.dumps(inventory, ensure_ascii=False, indent=2) + "\n"


def inventory_table(inventory):
    rows = ["# UI表示・操作条件の管理表", "",
            "ソースから自動生成。変更時は`PYTHONPATH=.:src .venv/bin/python scripts/inspect_ui_visibility.py --write`を実行する。",
            "品質チェックはソースとの不一致を検出する。詳細な条件・行番号・安定IDは`ui-visibility-inventory.json`を参照。",
            "実行時の未観測は非表示と断定しない。診断の使用方法と限界は[UI可視性管理](ui-visibility-management.md)を参照。", "",
            "| モジュール | 描画・表示候補 | 条件付き候補 | disabled指定 | expanded指定 | CSS非表示宣言 |", "|---|---:|---:|---:|---:|---:|"]
    for module in inventory["modules"]:
        rules = [row for row in inventory["rules"] if row["source"] == module["source"]]
        counts = [len(rules), sum(bool(row["guards"]) for row in rules),
                  sum("disabled" in row for row in rules), sum("expanded" in row for row in rules),
                  sum(row["kind"] == "presentation_rule" for row in rules)]
        rows.append(f"| `{module['source']}` | " + " | ".join(map(str, counts)) + " |")
    rows += ["", f"合計: {len(inventory['modules'])}モジュール、{len(inventory['rules'])}候補。", ""]
    return "\n".join(rows)


class _RenderObserver:
    def __init__(self, root, app, inventory):
        self.previous = sys.getprofile()
        self.by_file = {}
        for rule in inventory["rules"]:
            self.by_file.setdefault(str(root / rule["source"]), []).append(rule)
        self.snapshot = {"schema_version": 1, "app": app,
                         "catalog_sha256": hashlib.sha256(inventory_json(inventory).encode()).hexdigest(),
                         "unobserved_status": "unknown", "observed": {}}

    def __call__(self, frame, event, arg):
        if event != "call":
            return
        module = frame.f_globals.get("__name__", "")
        function = frame.f_locals.get("non_optional_func")
        command = getattr(function, "__name__", frame.f_code.co_name)
        if not module.startswith("streamlit.") or command not in UI_COMMANDS:
            return
        if frame.f_code.co_name != "wrapped_func" and command != "stop":
            return  # Count the public wrapper once, not its implementation too.
        caller = frame.f_back
        while caller is not None:
            candidates = self.by_file.get(caller.f_code.co_filename)
            if candidates:
                matches = [row for row in candidates
                           if row["line"] <= caller.f_lineno <= row["end_line"]
                           and row["command"] == command]
                if matches:
                    rule = min(matches, key=lambda row: row["end_line"] - row["line"])
                    observed = self.snapshot["observed"].setdefault(rule["id"], {
                        "source": rule["source"], "line": rule["line"], "command": command,
                        "call_attempts": 0, "disabled_true": 0, "disabled_false": 0,
                    })
                    observed["call_attempts"] += 1
                    options = frame.f_locals.get("kwargs", frame.f_locals)
                    disabled = options.get("disabled")
                    if isinstance(disabled, bool):
                        observed["disabled_true" if disabled else "disabled_false"] += 1
                    expanded = options.get("expanded")
                    if isinstance(expanded, bool):
                        observed["expanded"] = expanded
                    return
            caller = caller.f_back


def start_visibility_diagnostics(root: Path, app: str):
    """Opt in per script thread; never replace another debugger/profiler."""
    previous = sys.getprofile()
    if isinstance(previous, _RenderObserver):
        sys.setprofile(previous.previous)
    if os.environ.get(DIAGNOSTICS_ENV) != "1":
        return None
    if sys.getprofile() is not None:
        return {"schema_version": 1, "app": app, "status": "another_profiler_active"}
    observer = _RenderObserver(root.resolve(), app, build_visibility_inventory(root))
    sys.setprofile(observer)
    return observer.snapshot


def stop_visibility_diagnostics():
    observer = sys.getprofile()
    if isinstance(observer, _RenderObserver):
        sys.setprofile(observer.previous)
