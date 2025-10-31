import html
from typing import Any, Dict, List, Optional

import pandas as pd


def format_loss_texts(loss_texts, colors=None):
    """
    Concatenate a sequence of text-instance dicts into one HTML string,
    highlighting those with compute_loss=True by cycling through `colors`.

    Args:
        loss_texts (List[Dict[str, Any]]): Each dict must have:
            - "text": the string to render (may contain newlines, HTML will be escaped)
            - "compute_loss": bool, whether to highlight this segment.
        colors (List[str], optional): List of CSS color values for backgrounds.
            Defaults to a palette of 5 pastel colors.

    Returns:
        str: An HTML string with inline <span> wrappers.
    """
    # default pastel palette
    default_colors = [
        "#FFECB3",  # light amber
        "#C8E6C9",  # light green
        "#BBDEFB",  # light blue
        "#D1C4E9",  # lavender
        "#FFCDD2",  # light red
    ]
    palette = colors if (colors and len(colors) > 0) else default_colors

    html_parts = []
    highlight_idx = 0

    for inst in loss_texts:
        raw_text = inst.get("text", "")
        safe_text = html.escape(raw_text).replace("\n", "<br>")

        if inst.get("compute_loss", False):
            # pick current highlight color, then advance
            bg = palette[highlight_idx % len(palette)]
            highlight_idx += 1
            span = (
                f'<span style="background-color: {bg}; color: black;">'
                f"\n{safe_text}"
                "\n</span>"
            )
        else:
            # plain text in black (no background)
            span = f'<span style="color: black;">\n{safe_text}\n</span>'

        html_parts.append(span)

    # join without separator to preserve original flow
    return "".join(html_parts)


def format_texts_latex(
    loss_texts: List[Dict[str, Any]],
    colors: Optional[List[str]] = None,
    style_name: str = "plaincode",
    wrap_in_environment: bool = True,
    max_length: int = 4000,
) -> str:
    """
    Concatenate text-instance dicts into a LaTeX lstlisting block with per-span highlights.

    Each `inst` must have:
      - "text": str (may contain newlines)
      - "compute_loss": bool (highlight this span if True)

    Args:
        loss_texts: list of dicts as above.
        colors: list of xcolor background specs (e.g., ["yellow!35","green!30",...]).
                If None, uses a 5-color pastel palette.
        style_name: listings style to use (default "plaincode").
        wrap_in_environment: if True, wraps with \\begin{lstlisting}[style=...] ... \\end{lstlisting].
                             If False, returns only the inner content (useful for templating).

    Returns:
        str: LaTeX string ready for Overleaf.
    """
    # Default xcolor palette (backgrounds)
    # default_colors = ["yellow!35", "green!30", "cyan!30", "pink!35", "orange!35"]
    default_colors = ["hlAmber", "hlGreen", "hlBlue", "hlLavender", "hlRed"]

    palette = colors if (colors and len(colors) > 0) else default_colors

    def escape_for_hl_arg(s: str) -> str:
        """
        Escape LaTeX specials so the string can be safely used inside \hlc{...}.
        Non-highlighted text is left unescaped (listings prints it literally).
        """
        # IMPORTANT: do NOT HTML-escape here; we want raw text in LaTeX.
        repl = {
            "\\": r"\textbackslash{}",
            "{": r"\{",
            "}": r"\}",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "&": r"\&",
            "_": r"\_",
            "^": r"\textasciicircum{}",
            "~": r"\textasciitilde{}",
        }
        out = []
        for ch in s:
            out.append(repl.get(ch, ch))
        return "".join(out)

    parts: List[str] = []
    hi = 0
    has_loss_text = False

    for inst in loss_texts:
        raw_text = str(inst.get("text", ""))

        if inst.get("compute_loss", False):
            bg = palette[hi % len(palette)]
            hi += 1
            # We are exiting listings and injecting LaTeX; escape the content.
            safe = escape_for_hl_arg(raw_text)
            # replace newlines with \\
            safe = safe.replace("\n", "\\\\")
            # make sure doesn't end in solo \
            if safe.endswith("\\"):
                safe = safe[:-1]
            span = f"(*@\\hlc[{bg}]{{{safe}}}@*)"
            has_loss_text = True
        else:
            # Stay literal inside listings (no escaping, keep newlines verbatim)
            span = raw_text

        parts.append(span)
        # only add if max_length is not exceeded
        if max_length is not None:
            if len("".join(parts)) > max_length and has_loss_text:
                parts.append("...")
                break

    inner = "".join(parts)

    if wrap_in_environment:
        return (
            f"\\begin{{lstlisting}}[style={style_name}]\n{inner}\n\\end{{lstlisting}}\n"
        )
    else:
        return inner
