#!/bin/bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path

p = Path('/usr/local/lib/python3.12/dist-packages/prometheus_fastapi_instrumentator/routing.py')
if not p.exists():
    print('prometheus_fastapi_instrumentator routing.py not found, skipping')
    raise SystemExit(0)

text = p.read_text()
old = '''        if match == Match.FULL:
            route_name = route.path
            child_scope = {**scope, **child_scope}
            if isinstance(route, Mount) and route.routes:
                child_route_name = _get_route_name(child_scope, route.routes, route_name)
                if child_route_name is None:
                    route_name = None
                else:
                    route_name += child_route_name
            return route_name
        elif match == Match.PARTIAL and route_name is None:
            route_name = route.path
'''
new = '''        if match == Match.FULL:
            # FastAPI/Starlette newer versions can include internal router
            # entries that match but do not expose .path. Fall back to
            # .path_format or an empty route name instead of crashing every
            # request in prometheus metrics middleware.
            route_name = getattr(route, "path", getattr(route, "path_format", ""))
            child_scope = {**scope, **child_scope}
            if isinstance(route, Mount) and route.routes:
                child_route_name = _get_route_name(child_scope, route.routes, route_name)
                if child_route_name is None:
                    route_name = None
                else:
                    route_name += child_route_name
            return route_name
        elif match == Match.PARTIAL and route_name is None:
            route_name = getattr(route, "path", getattr(route, "path_format", ""))
'''

if new in text:
    print('Prometheus FastAPI routing compatibility patch already applied:', p)
elif old in text:
    p.write_text(text.replace(old, new, 1))
    print('Patched Prometheus FastAPI routing compatibility:', p)
else:
    raise SystemExit('Could not find expected prometheus_fastapi_instrumentator routing block')
PY

python3 -m py_compile /usr/local/lib/python3.12/dist-packages/prometheus_fastapi_instrumentator/routing.py
