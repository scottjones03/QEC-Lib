"""
Video / file rendering for QCCD transport animations.

Provides ``render_animation()`` — encodes a ``FuncAnimation`` to MP4
via ffmpeg and optionally displays inline in a Jupyter notebook.
Falls back to ``to_jshtml()`` if ffmpeg is unavailable.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Any, Optional

from .display import _require_matplotlib


def render_animation(
    anim: Any,
    output_path: Optional[str] = None,
    *,
    fps: int = 5,
    dpi: int = 120,
    bitrate: int = 2000,
    writer: str = "ffmpeg",
    display_inline: bool = True,
    width: int = 900,
    fallback_to_jshtml: bool = False,
    jshtml_embed_limit_mb: int = 50,
) -> Optional[Path]:
    """Render a FuncAnimation to MP4 video and optionally show inline.

    Replaces the common notebook pattern of ``anim.to_jshtml()`` +
    ``display(HTML(...))``, which breaks on large animations that exceed
    matplotlib's embed-limit.  Instead, encodes to H.264 MP4 via ffmpeg
    and embeds the video using a compact HTML5 ``<video>`` tag.

    Parameters
    ----------
    anim : matplotlib.animation.FuncAnimation
        Animation object returned by :func:`animate_transport`.
    output_path : str | pathlib.Path | None
        Where to write the ``.mp4`` file.  If *None*, defaults to
        ``_anim_output.mp4`` in the current directory.
    fps : int
        Frames per second (default 5).
    dpi : int
        Resolution (default 120).
    bitrate : int
        Video bitrate in kbps (default 2000).
    writer : str
        Matplotlib animation writer name (default ``'ffmpeg'``).
    display_inline : bool
        If *True*, embed the video inline in a Jupyter notebook.
    width : int
        Width of the inline ``<video>`` element (pixels).
    fallback_to_jshtml : bool
        Fall back to ``to_jshtml()`` if encoding fails.
    jshtml_embed_limit_mb : int
        Embed limit (MB) when falling back to jshtml.

    Returns
    -------
    pathlib.Path | None
        Path to the rendered video file, or *None* if jshtml fallback
        was used.

    Raises
    ------
    RuntimeError
        If encoding fails and ``fallback_to_jshtml`` is False.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = Path("_anim_output.mp4")
    else:
        output_path = Path(output_path)

    t0 = _time.perf_counter()

    try:
        print(
            f"Encoding animation \u2192 {output_path} "
            f"(writer={writer}, fps={fps}, dpi={dpi}, bitrate={bitrate}) \u2026"
        )
        from .constants import BG_COLOR

        savefig_kw: dict = {"facecolor": BG_COLOR}
        fig_obj = getattr(anim, "_fig", None)
        if fig_obj is not None:
            savefig_kw["facecolor"] = fig_obj.get_facecolor()

        anim.save(
            str(output_path),
            writer=writer,
            fps=fps,
            dpi=dpi,
            bitrate=bitrate,
            savefig_kwargs=savefig_kw,
        )
        t1 = _time.perf_counter()
        fsize_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Saved in {t1 - t0:.1f}s \u2014 {fsize_mb:.1f} MB")

    except Exception as exc:
        if fallback_to_jshtml:
            import warnings
            import matplotlib

            warnings.warn(
                f"Video encoding failed ({exc}); falling back to jshtml. "
                f"Install ffmpeg for better results: brew install ffmpeg",
                stacklevel=2,
            )
            matplotlib.rcParams["animation.embed_limit"] = jshtml_embed_limit_mb
            html_str = anim.to_jshtml(default_mode="loop")
            if display_inline:
                from IPython.display import HTML, display as _display

                _display(HTML(html_str))
            print(f"Fallback jshtml: {len(html_str) // 1024} KB")
            if fig_obj is not None:
                plt.close(fig_obj)
            return None
        else:
            raise RuntimeError(
                f"Animation encoding failed: {exc}\n"
                f"Install ffmpeg (brew install ffmpeg) or set "
                f"fallback_to_jshtml=True."
            ) from exc

    if display_inline:
        import base64

        from IPython.display import HTML, display as _display

        video_b64 = base64.b64encode(output_path.read_bytes()).decode("ascii")
        mime = "video/mp4"
        if output_path.suffix.lower() == ".gif":
            mime = "image/gif"
        _display(
            HTML(
                f'<video controls autoplay loop width="{width}" '
                f'style="max-width:100%">'
                f'<source src="data:{mime};base64,{video_b64}" '
                f'type="{mime}">'
                f"Your browser does not support HTML5 video.</video>"
            )
        )

    if fig_obj is not None:
        plt.close(fig_obj)

    print(f"\u2705 Animation: {fsize_mb:.1f} MB \u2014 {output_path}")
    return output_path


__all__ = ["render_animation"]
