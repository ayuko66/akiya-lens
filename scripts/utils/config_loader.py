from __future__ import annotations
from pathlib import Path
import re
from string import Template
from datetime import datetime
import yaml

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Backward-compat alias used by some scripts
def load_config(path: str | Path) -> dict:
    return load_yaml(path)

def render_filename(template: str, project: dict) -> str:
    """テンプレートに project.version と date を展開する。

    サポート: ${project.version}, ${date:%Y%m%d}
    string.Template は '.' を含む識別子をサポートしないため、
    project.version は明示的に置換し、date は正規表現で処理する。
    """
    s = template
    # ${project.version} の置換（ドットを含むため Template は使わない）
    s = re.sub(r"\$\{project\.version\}", project.get("version", "v1"), s)

    # ${date:%...} の置換（% を含むフォーマット全体を渡す）
    def sub_date(m):
        fmt = m.group(1)
        # group(1) は % を含まないため先頭に付与
        return datetime.now().strftime("%" + fmt)

    s = re.sub(r"\$\{date:%([^}]+)\}", sub_date, s)
    return s
