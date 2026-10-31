"""Temporary CI diagnostic: run pytest and surface failures as check annotations."""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

JUNIT = ".ci-junit.xml"


def escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", f"--junitxml={JUNIT}"],
    check=False,
)
if result.returncode != 0 and os.path.exists(JUNIT):
    root = ET.parse(JUNIT).getroot()
    for case in root.iter("testcase"):
        for tag in ("failure", "error"):
            node = case.find(tag)
            if node is None:
                continue
            message = (node.get("message") or "")[:800]
            detail = (node.text or "")[:2500]
            print(
                f"::error title={escape(case.get('name', 'test'))}::"
                f"{escape(message + chr(10) + detail)}"
            )
sys.exit(result.returncode)
