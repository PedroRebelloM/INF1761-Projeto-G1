from __future__ import annotations

def readfile (filename: str) -> str:
  """Reads `filename` and returns its contents as a string; no path
  resolution or include-handling."""
  with open(filename) as f:
    lines = f.readlines()
  text = ""
  for l in lines:
    text = text + l
  return text
