#!/usr/bin/env python3
"""Render a privacy-safe repository metrics SVG from GitHub-owned data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


API = "https://api.github.com"
API_VERSION = "2022-11-28"


def _request_json(path: str, token: str, *, accept: str = "application/vnd.github+json") -> tuple[Any, Any]:
    request = Request(
        f"{API}{path}",
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "User-Agent": "repository-metrics-renderer",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed GitHub API origin
        return json.load(response), response.headers


def fetch_snapshot(repository: str, token: str) -> dict[str, Any]:
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required when --snapshot is not supplied")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("repository must use owner/name format")

    encoded = "/".join(quote(part, safe="") for part in repository.split("/", 1))
    repo, _ = _request_json(f"/repos/{encoded}", token)
    views, _ = _request_json(f"/repos/{encoded}/traffic/views", token)
    clones, _ = _request_json(f"/repos/{encoded}/traffic/clones", token)
    _, commit_headers = _request_json(f"/repos/{encoded}/commits?per_page=1", token)

    link = commit_headers.get("Link", "")
    last_page = re.search(r"[?&]page=(\d+)[^>]*>; rel=\"last\"", link)
    commit_count = int(last_page.group(1)) if last_page else 1

    starred_at: list[str] = []
    page = 1
    while True:
        entries, _ = _request_json(
            f"/repos/{encoded}/stargazers?per_page=100&page={page}",
            token,
            accept="application/vnd.github.star+json",
        )
        starred_at.extend(
            item["starred_at"]
            for item in entries
            if isinstance(item, dict) and isinstance(item.get("starred_at"), str)
        )
        if len(entries) < 100:
            break
        page += 1

    return {
        "repository": repository,
        "created_at": repo["created_at"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "stars": int(repo["stargazers_count"]),
        "forks": int(repo["forks_count"]),
        "commits": commit_count,
        "unique_visitors_14d": int(views["uniques"]),
        "views_14d": int(views["count"]),
        "unique_cloners_14d": int(clones["uniques"]),
        "clones_14d": int(clones["count"]),
        "starred_at": sorted(starred_at),
    }


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _compact(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def render_svg(snapshot: dict[str, Any]) -> str:
    repository = escape(str(snapshot["repository"]))
    created = _parse_time(str(snapshot["created_at"]))
    generated = _parse_time(str(snapshot["generated_at"]))
    if generated <= created:
        generated = created.replace(microsecond=0)

    stars = max(0, int(snapshot["stars"]))
    star_dates = sorted(
        date for date in (_parse_time(str(value)) for value in snapshot.get("starred_at", []))
        if created <= date <= generated
    )

    chart_x, chart_y, chart_w, chart_h = 64.0, 150.0, 540.0, 292.0
    baseline = chart_y + chart_h
    seconds = max((generated - created).total_seconds(), 1.0)
    y_max = max(stars, 1)

    def x_for(date: datetime) -> float:
        return chart_x + chart_w * max(0.0, min(1.0, (date - created).total_seconds() / seconds))

    def y_for(value: int) -> float:
        return baseline - chart_h * max(0.0, min(1.0, value / y_max))

    path = [f"M {chart_x:.1f} {y_for(0):.1f}"]
    count = 0
    for date in star_dates:
        x = x_for(date)
        path.append(f"L {x:.1f} {y_for(count):.1f}")
        count += 1
        path.append(f"L {x:.1f} {y_for(count):.1f}")
    if count < stars:
        path.append(f"L {chart_x + chart_w:.1f} {y_for(stars):.1f}")
    else:
        path.append(f"L {chart_x + chart_w:.1f} {y_for(count):.1f}")
    line_path = " ".join(path)
    area_path = f"{line_path} L {chart_x + chart_w:.1f} {baseline:.1f} Z"

    ticks = sorted({0, max(1, math.ceil(y_max / 2)), y_max})
    grid = []
    for value in ticks:
        y = y_for(value)
        grid.append(
            f'<line x1="{chart_x:.1f}" y1="{y:.1f}" x2="{chart_x + chart_w:.1f}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{chart_x - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" class="axis">{value}</text>'
        )

    metrics = (
        ("Stars", stars),
        ("Forks", int(snapshot["forks"])),
        ("Commits", int(snapshot["commits"])),
        ("Unique visitors · 14d", int(snapshot["unique_visitors_14d"])),
        ("Unique cloners · 14d", int(snapshot["unique_cloners_14d"])),
        ("Total clones · 14d", int(snapshot["clones_14d"])),
    )
    cards = []
    for index, (label, value) in enumerate(metrics):
        column, row = index % 2, index // 2
        x, y = 650 + column * 142, 154 + row * 104
        label_parts = label.split(" · ", 1)
        label_svg = f'<text x="15" y="25" class="metric-label">{escape(label_parts[0])}</text>'
        if len(label_parts) == 2:
            label_svg += f'<text x="15" y="40" class="metric-period">{escape(label_parts[1])}</text>'
        cards.append(
            f'<g transform="translate({x} {y})">'
            '<rect width="128" height="84" rx="17" class="card-shadow"/>'
            '<rect width="128" height="84" rx="17" class="card"/>'
            f'{label_svg}'
            f'<text x="15" y="70" class="metric-value">{escape(_compact(value))}</text>'
            '</g>'
        )

    empty_note = ""
    if stars == 0:
        empty_note = '<text x="334" y="292" text-anchor="middle" class="empty">Waiting for the first star</text>'

    updated = generated.strftime("%Y-%m-%d UTC")
    start_label = created.strftime("%Y-%m")
    end_label = generated.strftime("%Y-%m")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="560" viewBox="0 0 960 560" role="img" aria-labelledby="title desc">
  <title id="title">{repository} repository metrics</title>
  <desc id="desc">Star growth curve with stars, forks, commits, visitors, cloners, and clone totals.</desc>
  <style>
    .background {{ fill: #ffffff; }}
    .outline {{ fill: none; stroke: #171717; stroke-width: 2.4; }}
    .title {{ font: 700 30px ui-rounded, "Comic Sans MS", system-ui, sans-serif; fill: #171717; }}
    .subtitle {{ font: 500 15px ui-rounded, "Comic Sans MS", system-ui, sans-serif; fill: #666666; }}
    .section {{ font: 700 18px ui-rounded, "Comic Sans MS", system-ui, sans-serif; fill: #252525; }}
    .grid {{ stroke: #dedede; stroke-width: 1.2; stroke-dasharray: 5 7; }}
    .axis {{ font: 12px ui-rounded, "Comic Sans MS", system-ui, sans-serif; fill: #777777; }}
    .area {{ fill: #dff7e5; opacity: .76; }}
    .curve-echo {{ fill: none; stroke: #171717; stroke-width: 5.6; opacity: .12; stroke-linejoin: round; }}
    .curve {{ fill: none; stroke: #18a558; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }}
    .card-shadow {{ fill: #ececec; transform: translate(3px, 4px); }}
    .card {{ fill: #ffffff; stroke: #222222; stroke-width: 2; }}
    .metric-label {{ font: 650 11px ui-rounded, "Comic Sans MS", system-ui, sans-serif; fill: #555555; }}
    .metric-period {{ font: 600 9.5px ui-rounded, "Comic Sans MS", system-ui, sans-serif; fill: #888888; }}
    .metric-value {{ font: 800 28px ui-rounded, "Comic Sans MS", system-ui, sans-serif; fill: #171717; }}
    .empty {{ font: 600 16px ui-rounded, "Comic Sans MS", system-ui, sans-serif; fill: #888888; }}
    .footer {{ font: 12px ui-rounded, "Comic Sans MS", system-ui, sans-serif; fill: #666666; }}
  </style>
  <rect class="background" width="960" height="560" rx="24"/>
  <rect class="outline" x="12" y="12" width="936" height="536" rx="22"/>
  <text x="54" y="61" class="title">Repository Pulse</text>
  <text x="54" y="88" class="subtitle">{repository} · privacy-safe public activity</text>
  <path d="M 844 47 l 8 16 18 3 -13 12 3 18 -16 -9 -16 9 3 -18 -13 -12 18 -3 z" fill="#18a558" opacity=".9"/>
  <text x="64" y="128" class="section">Stars over time</text>
  {''.join(grid)}
  <line x1="{chart_x:.1f}" y1="{baseline:.1f}" x2="{chart_x + chart_w:.1f}" y2="{baseline:.1f}" class="outline"/>
  <path d="{area_path}" class="area"/>
  <path d="{line_path}" class="curve-echo"/>
  <path d="{line_path}" class="curve"/>
  {empty_note}
  <text x="{chart_x:.1f}" y="{baseline + 24:.1f}" class="axis">{start_label}</text>
  <text x="{chart_x + chart_w:.1f}" y="{baseline + 24:.1f}" text-anchor="end" class="axis">{end_label}</text>
  {''.join(cards)}
  <text x="54" y="520" class="footer">Auto-refreshed on stars and weekly · GitHub Traffic uses the rolling 14-day owner view · Updated {updated}</text>
</svg>
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, help="GitHub owner/name")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path, help="Render from a local JSON snapshot instead of the API")
    args = parser.parse_args()

    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text())
    else:
        snapshot = fetch_snapshot(args.repository, os.environ.get("GITHUB_TOKEN", ""))
    if snapshot.get("repository") != args.repository:
        raise ValueError("snapshot repository does not match --repository")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(snapshot))
    print(f"rendered {args.output} for {args.repository}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
