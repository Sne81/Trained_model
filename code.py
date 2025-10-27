"""
Minimal placeholder module named 'code' to avoid shadowing the standard library.
This file intentionally provides a tiny InteractiveConsole implementation so that
imports of the standard library `code` module (for example via `pdb`) don't fail
when a local file named `code.py` exists in the working directory.
"""

class InteractiveConsole:
    def __init__(self, locals=None):
        self.locals = locals or {}

    def push(self, line):
        try:
            exec(line, self.locals)
        except Exception:
            # intentionally swallow exceptions here
            pass
        return False


class InteractiveInterpreter(InteractiveConsole):
    pass
