"""Day 31: write the HTTP API reference from the app's own OpenAPI schema.

    python -m app.export_api_docs            # writes docs/api.md

Generated rather than written by hand so that it cannot drift from the code:
every endpoint FastAPI serves, with the request and response models it
declares and their fields. The file is documentation for people and for the
developer assistant's index alike - run this, then re-index.
"""

import argparse
from pathlib import Path

from app.main import app

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "api.md"


def _ref(schema: dict | None) -> str:
    if not schema:
        return ""

    if "$ref" in schema:
        return schema["$ref"].split("/")[-1]

    for key in ("anyOf", "allOf", "oneOf"):
        if key in schema:
            names = [_ref(part) or part.get("type", "") for part in schema[key]]
            return " | ".join(name for name in names if name)

    if schema.get("type") == "array":
        return f"list[{_ref(schema.get('items')) or schema.get('items', {}).get('type', 'any')}]"

    return schema.get("type", "")


def _fields(name: str, components: dict) -> list[str]:
    model = components.get(name)

    if not model or "properties" not in model:
        return []

    required = set(model.get("required", []))
    lines = []

    for field, schema in model["properties"].items():
        kind = _ref(schema) or "any"
        flag = "required" if field in required else "optional"
        lines.append(f"- `{field}`: {kind} ({flag})")

    return lines


def _group(path: str) -> str:
    first = path.strip("/").split("/")[0]

    return f"/{first}"


def render() -> str:
    spec = app.openapi()
    components = spec.get("components", {}).get("schemas", {})
    groups: dict[str, list[str]] = {}

    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            lines = [f"### {method.upper()} {path}", ""]
            description = (operation.get("description") or operation.get("summary") or "").strip()

            if description:
                lines += [description.split("\n\n")[0].replace("\n", " "), ""]

            parameters = [
                f"- `{item['name']}` ({item['in']}): {_ref(item.get('schema')) or 'any'}"
                for item in operation.get("parameters", [])
            ]

            if parameters:
                lines += ["Parameters:", *parameters, ""]

            body = (
                operation.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
            )

            if body:
                request = _ref(body)
                lines += [f"Request body: `{request}`", *_fields(request, components), ""]

            ok = operation.get("responses", {}).get("200", {}).get("content", {}).get("application/json", {})
            response = _ref(ok.get("schema"))

            if response:
                lines += [f"Response 200: `{response}`", *_fields(response, components), ""]

            groups.setdefault(_group(path), []).append("\n".join(lines))

    out = [
        "# HTTP API reference",
        "",
        "Generated from the FastAPI app's OpenAPI schema by `python -m app.export_api_docs`.",
        "Do not edit by hand - change the code and regenerate.",
        "",
        f"{sum(len(items) for items in groups.values())} endpoints. Interactive version: `/docs` on a running server.",
        "",
    ]

    for group in sorted(groups):
        out += [f"## {group}", "", *groups[group]]

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.export_api_docs", description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUTPUT)
    options = parser.parse_args(argv)
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(render(), encoding="utf-8")
    print(f"Written: {options.out}")


if __name__ == "__main__":
    main()
