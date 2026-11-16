"""supercar-bench MCP tool server.

Exposes the CAR-bench in-car assistant toolset over the Model Context
Protocol so any agent (not only this repository's harness) can call it.

Design notes
------------
* The 58 tool classes are used unmodified. They read their state through the
  ``context_state`` / ``fixed_context`` contextvars, so the server installs a
  per-session state pair around every ``tools/call`` and the tool code needs
  no changes.
* One session owns one ContextState, one FixedContext, one tool allowlist and
  the error lists the tools append to. Sessions are independent; a lock
  serializes calls within a session, which keeps concurrent writes to the
  same state ordered.
* ``tools/list`` converts each tool's OpenAI function schema
  (``function.parameters``) into the MCP shape (top-level ``inputSchema``).
"""
