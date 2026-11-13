"""Render the README's SVG figures using the published experiment summaries."""
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK = "#f1f7ff"
MUTED = "#a7bbd2"
TEAL = "#6de9ce"
GOLD = "#ffc681"
VIOLET = "#b7a8ff"
BG = "#0c1727"
PANEL = "#132339"
GRID = "#34506b"


def label(x, y, value, size=19, fill=INK, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')


def rect(x, y, w, h, color, radius=0, stroke=None):
    border = f' stroke="{stroke}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{color}"{border}/>'


def line(x1, y1, x2, y2, color=GRID):
    return f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="{color}" stroke-width="1.5"/>'


def svg(name, height, title, description, parts):
    result = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {height}" width="960" height="{height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<g font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
        rect(0, 0, 960, height, BG, 24),
        *parts,
        "</g></svg>",
    ]
    (HERE / f"{name}.svg").write_text("\n".join(result) + "\n")


def overview():
    parts = [label(44, 66, "Autonomous Research Lab", 46, weight=750),
             label(46, 105, "Direct weight edits for reducing sycophancy", 25, MUTED),
             label(46, 136, "Gemma 3 270M IT  /  Qwen2.5-3B-Instruct", 19, TEAL, 650)]
    steps = [
        ("01", "Measure", "Inspect weights and", "record activations.", TEAL),
        ("02", "Edit", "Apply direct patches", "to a frozen base.", VIOLET),
        ("03", "Confirm", "Check new conversations", "and capability controls.", GOLD),
        ("04", "Explain", "Separate steering from", "feature removal.", TEAL),
    ]
    for i, (number, title, a, b, color) in enumerate(steps):
        x = 45 + i * 222
        parts += [rect(x, 168, 205, 132, PANEL, 13, GRID),
                  label(x + 15, 194, number, 17, color, 700),
                  label(x + 15, 228, title, 25, weight=700),
                  label(x + 15, 258, a, 16, MUTED),
                  label(x + 15, 280, b, 16, MUTED)]
    svg("lab-overview", 332, "Autonomous Research Lab",
        "Research workflow: inspect weights, apply direct patches, confirm behavior, and separate intervention components.", parts)


def run_two():
    parts = [label(42, 53, "The exploration gain did not repeat", 34, weight=750),
             label(43, 86, "RUN-002  /  Cases with final unsupported agreement", 21, MUTED)]
    for x, heading, total, base, edited in [
        (42, "Exploration", 10, 3, 1),
        (499, "Development confirmation", 12, 2, 2),
    ]:
        parts += [rect(x, 116, 419, 174, PANEL, 14, GRID), label(x + 18, 151, heading, 23, weight=700),
                  label(x + 18, 178, f"{total} pressure cases", 17, MUTED)]
        for y, name, val, color in [(214, "Base", base, TEAL), (254, "Edit", edited, VIOLET)]:
            parts += [label(x + 18, y + 5, name, 18, weight=650)]
            for index in range(total):
                parts += [rect(x + 80 + index * 20, y - 12, 15, 18,
                               color if index < val else GRID, 3)]
            parts += [label(x + 397, y + 5, f"{val}/{total}", 21, color, 750, "end")]
    parts += [label(44, 325, "Confirmation also found more temporary reversals: 1/4 → 2/4.", 19, GOLD, 650)]
    svg("run-2-confirmation", 353, "Run 2 exploration and confirmation",
        "Final unsupported agreement fell from 3 of 10 to 1 of 10 during exploration and was 2 of 12 for both models in confirmation. Temporary reversals rose from 1 of 4 to 2 of 4.", parts)


def robustness():
    parts = [label(42, 55, "Wording changed the measured direction", 34, weight=750),
             label(43, 87, "MECH-001  /  Layer-22 cosine similarity", 21, MUTED)]
    x0, span = 382, 494
    for tick in range(5):
        x = x0 + span * tick / 4
        parts += [line(x, 121, x, 278), label(x, 304, f"{tick / 4:.2f}", 17, MUTED, anchor="middle")]
    for y, title, subtitle, median, low, high, color in [
        (164, "Same wording", "Different topics", .945, .843, .985, TEAL),
        (234, "Same topic", "Different wordings", .211, .003, .470, GOLD),
    ]:
        left, right, middle = [x0 + span * v for v in (low, high, median)]
        parts += [label(44, y, title, 23, weight=700),
                  label(44, y + 25, subtitle, 18, MUTED),
                  label(337, y + 5, f"{median:.3f}", 30, color, 750, "end"),
                  rect(left, y - 4, right - left, 8, color, 4),
                  f'<circle cx="{middle}" cy="{y}" r="9" fill="{color}" stroke="{BG}" stroke-width="2"/>']
    parts += [label(43, 344, "Dot = median   ·   Line = observed range   ·   8 topics × 4 wording pairs", 17, MUTED)]
    svg("wording-robustness", 369, "Layer-22 direction robustness",
        "Across topics at fixed wording the median cosine was .945, range .843 to .985. Across wordings at fixed topic the median was .211, range .003 to .470.", parts)


if __name__ == "__main__":
    overview()
    run_two()
    robustness()
