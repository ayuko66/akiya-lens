from __future__ import annotations
from pathlib import Path
import re
from string import Template
from datetime import datetime
import yaml

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def render_filename(template: str, project: dict) -> str:
    t = Template(template)
    s = t.safe_substitute({"project.version": project.get("version", "v1")})
    def sub_date(m):
        fmt = m.group(1)
        return datetime.now().strftime(fmt)
    s = re.sub(r"\$\{date:%([^}]+)\}", sub_date, s)
    return s
