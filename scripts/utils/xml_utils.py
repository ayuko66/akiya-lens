# scripts/utils/xml_utils.py
# -*- coding: utf-8 -*-
"""
Generic XML parse utilities for KSJ and similar datasets.
Design goals:
- Minimal deps (stdlib + pandas)
- Reusable: field mapping is driven by YAML (xpath per field)
- Namespaces: define in YAML and pass to extractor
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterable
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

@dataclass
class XPathField:
    name: str
    xpath: str
    post: Optional[str] = None  # optional post-processing: "int","float","strip"

def _post_process(val: Optional[str], post: Optional[str]) -> Any:
    if val is None:
        return None
    if post is None:
        return val
    v = val.strip()
    if post == "int":
        v = v.replace(",", "")
        return int(v) if v != "" else None
    if post == "float":
        v = v.replace(",", "")
        return float(v) if v != "" else None
    if post == "strip":
        return v
    return v

def iter_records(xml_path: Path, item_xpath: str, fields: List[XPathField], ns: Dict[str, str]) -> Iterable[Dict[str, Any]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for item in root.findall(item_xpath, ns):
        rec: Dict[str, Any] = {}
        for f in fields:
            node = item.find(f.xpath, ns) if f.xpath else None
            text = node.text if (node is not None and node.text is not None) else None
            rec[f.name] = _post_process(text, f.post)
        rec["source_file"] = str(xml_path)
        yield rec

def to_dataframe(recs: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(recs))
    return df
