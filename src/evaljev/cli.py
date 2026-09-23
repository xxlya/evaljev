"""``evaljev`` — turn a trace file into a monitoring page, from the command line.

Three commands, and the whole point is that none of them need a service, a key or
a network call:

    evaljev console traces.jsonl                 # what needs a person, and where it was flagged
    evaljev serve traces.jsonl                   # the same page, refreshing as traffic arrives
    evaljev demo                                 # the page, on a real recorded run

Everything the console does not show is still in the report it is built from:
``--json report.json`` writes the whole thing.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import webbrowser
from collections.abc import Sequence
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from .models import DecisionTrace
from .report import build_report, render_html
from .store import JsonlTraceStore, latest_by_trace_id

DEFAULT_TRACES = "traces.jsonl"


def _load(paths: Sequence[str]) -> list[DecisionTrace]:
    """Read one or more append-only trace files into the newest version of each trace."""
    traces: list[DecisionTrace] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"evaljev: no such trace file: {path}")
        traces.extend(JsonlTraceStore(p).list())
    if not traces:
        raise SystemExit(
            f"evaljev: {', '.join(paths)} contained no traces. Record some with "
            "Monitor(JsonlTraceStore(...)).run(...) first."
        )
    return latest_by_trace_id(traces)


def _sample_traces() -> list[DecisionTrace]:
    """The recorded run bundled with the package, for ``evaljev demo``.

    The three-node support assistant in ``examples/support_assistant.py``, run
    against the live API — real decisions, real probabilities, real latencies, and
    two real edits to one question part-way through. A simulated dashboard would be
    the one thing this library exists to argue against.
    """
    blob = resources.files("evaljev").joinpath("assets/sample-traces.jsonl.gz").read_bytes()
    lines = gzip.decompress(blob).decode("utf-8").splitlines()
    return latest_by_trace_id(
        DecisionTrace.model_validate(json.loads(line)) for line in lines if line.strip()
    )


def _report_kwargs(args) -> dict:
    return {
        "workflow_id": args.workflow,
        "title": args.title,
        "unsure_below": args.unsure_below,
        "human_actions": (
            [a.strip() for a in args.human_actions.split(",") if a.strip()]
            if args.human_actions
            else None
        ),
        "window_size": args.window,
        "recent": args.recent,
    }


def _write(report, out: Path, json_path: str | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(report), encoding="utf-8")
    if json_path:
        Path(json_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def _summarize(report, out: Path) -> None:
    queue = report["queue"]
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
    print(f"  {queue['headline']}")
    for row in queue["by_flag"]:
        print(f"  {row['n']:>4}  {row['label']}")


def cmd_report(args) -> int:
    report = build_report(_load(args.paths), **_report_kwargs(args))
    out = Path(args.output)
    _write(report, out, args.json)
    _summarize(report, out)
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def cmd_demo(args) -> int:
    report = build_report(
        _sample_traces(),
        title=args.title or "Support assistant — decision health",
        unsure_below=args.unsure_below,
        window_size=args.window,
        recent=args.recent,
        workflow_id=args.workflow,
        human_actions=_report_kwargs(args)["human_actions"],
    )
    out = Path(args.output)
    _write(report, out, args.json)
    _summarize(report, out)
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


class _Handler(BaseHTTPRequestHandler):
    """Rebuilds the report per request, so the page tracks the file as it grows."""

    def __init__(self, *a, paths, kwargs, **kw):
        self._paths, self._kwargs = paths, kwargs
        super().__init__(*a, **kw)

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _build(self):
        return build_report(_load(self._paths), **self._kwargs)

    def do_GET(self) -> None:  # stdlib naming
        try:
            if self.path.startswith("/report.json"):
                self._send(json.dumps(self._build(), default=str).encode(), "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(render_html(self._build()).encode(), "text/html; charset=utf-8")
            else:
                self.send_error(404)
        except SystemExit as exc:  # an empty or missing file, mid-run
            self.send_error(503, str(exc))
        except Exception as exc:  # noqa: BLE001 — a dev server should say what broke
            self.send_error(500, f"{type(exc).__name__}: {exc}")

    def log_message(self, fmt, *a):  # keep the console readable
        pass


def cmd_serve(args) -> int:
    kwargs = _report_kwargs(args)
    kwargs["live_interval_ms"] = int(args.refresh * 1000)
    _load(args.paths)  # fail fast, before binding the port
    handler = partial(_Handler, paths=args.paths, kwargs=kwargs)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{'localhost' if args.host in ('', '0.0.0.0') else args.host}:{args.port}/"
    print(f"EvalJev dashboard on {url}")
    print(f"  reading {', '.join(args.paths)} · refreshing every {args.refresh:g}s · ctrl-c to stop")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workflow", help="only include traces from this workflow_id")
    p.add_argument("--title", help="page title (default: the workflow id)")
    p.add_argument(
        "--unsure-below",
        type=float,
        default=0.6,
        metavar="P",
        help="probability below which a decision counts as one a human should see "
        "(default: 0.6). A probability, never the API's `confidence`, which rescales "
        "with the number of options.",
    )
    p.add_argument(
        "--window", type=int, metavar="N",
        help="decisions per comparison window (default: a quarter of each decision point's traffic)",
    )
    p.add_argument("--recent", type=int, default=60, metavar="N",
                   help="how many recent decisions to list (default: 60)")
    p.add_argument(
        "--human-actions",
        metavar="A,B",
        help="the branches that mean a person is now involved, comma separated "
        "(default: a small vocabulary matched by name, e.g. human, needs_review, escalate). "
        "This is what separates 'the workflow answered alone and should not have' from "
        "'the workflow escalated it, as designed'.",
    )
    p.add_argument("--json", metavar="FILE", help="also write the report as JSON")
    p.add_argument("--open", action="store_true", help="open the page in a browser")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaljev",
        description="Runtime assurance for typed decision workflows. "
        "Turns recorded decisions into a monitoring page you can read.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    console = sub.add_parser(
        "console", help="which requests need a person, and where they were flagged"
    )
    console.add_argument("paths", nargs="*", default=[DEFAULT_TRACES], metavar="TRACES",
                         help=f"one or more .jsonl trace files (default: {DEFAULT_TRACES})")
    console.add_argument("-o", "--output", default="console.html", metavar="FILE",
                         help="where to write the page (default: console.html)")
    _add_common(console)
    console.set_defaults(func=cmd_report)

    serve = sub.add_parser("serve", help="serve the page locally and refresh it as traces arrive")
    serve.add_argument("paths", nargs="*", default=[DEFAULT_TRACES], metavar="TRACES")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--refresh", type=float, default=5.0, metavar="SECONDS",
                       help="how often the open page re-reads the traces (default: 5)")
    _add_common(serve)
    serve.set_defaults(func=cmd_serve, open=False)

    demo = sub.add_parser(
        "demo", help="render the page from the example workflow's recorded run"
    )
    demo.add_argument("-o", "--output", default="evaljev-demo.html", metavar="FILE")
    _add_common(demo)
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
