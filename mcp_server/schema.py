"""Tool schema conversion between OpenAI function calling and MCP."""

from typing import Any, Dict, List


def mcp_tool_schema(tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one ``get_info()`` payload into an MCP tool definition.

    ``get_info()`` returns the OpenAI function calling shape::

        {"type": "function", "function": {"name", "description", "parameters"}}

    MCP expects the name and description at the top level and the JSON Schema
    under ``inputSchema``. The parameter schema itself is already JSON Schema
    and is passed through unchanged.
    """
    function = tool_info.get("function", tool_info)
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    return {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "inputSchema": parameters,
    }


def mcp_tools_list(tools_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a list of ``get_info()`` payloads into MCP tool definitions."""
    return [mcp_tool_schema(info) for info in tools_info]
