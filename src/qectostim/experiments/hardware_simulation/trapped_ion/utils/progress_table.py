"""
Progress table widget for parallel WISE config execution.

Displays a live-updating table of worker progress using ipywidgets
in Jupyter notebooks, with fallback to tqdm for terminal execution.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import ctypes
import time
import threading


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Memory State for Cross-Process Progress Updates
# ═══════════════════════════════════════════════════════════════════════════════

# Status codes for worker slots
STATUS_IDLE = 0
STATUS_RUNNING = 1
STATUS_DONE = 2
STATUS_ERROR = 3

# Message buffer size (bytes)
MSG_SIZE = 128


@dataclass
class SharedProgressSlot:
    """One worker's progress state using Manager proxy objects.
    
    Uses Manager-based proxy objects (dict/list) which ARE picklable
    and can be sent to child processes via spawn context (macOS default).
    
    The slot tracks THREE progress bars per worker:
    - **Route progress** (route_current/route_total): MS rounds
    - **Patch progress** (patch_current/patch_total): patches within a tiling cycle
    - **SAT progress** (sat_current/sat_total): SAT configs within current patch
    
    Keys in the proxy dict:
        route_current, route_total, route_message  — MS round progress
        patch_current, patch_total, patch_message  — patch-within-cycle progress
        sat_current, sat_total, sat_message        — SAT config progress
        status, config_id                          — overall slot state
    """
    _data: Any  # Manager proxy dict
    
    @classmethod
    def create(cls, manager: mp.managers.SyncManager) -> "SharedProgressSlot":
        """Create a new slot with Manager-based shared state."""
        data = manager.dict({
            # Route progress: MS rounds
            'route_current': 0,
            'route_total': 1,
            'route_message': '',
            # Patch progress: patches within tiling cycle
            'patch_current': 0,
            'patch_total': 1,
            'patch_message': '',
            # SAT progress: configs within current patch
            'sat_current': 0,
            'sat_total': 1,
            'sat_message': '',
            # Slot state
            'status': STATUS_IDLE,
            'config_id': -1,
        })
        return cls(_data=data)
    
    # ─── Route (MS round) progress ───
    @property
    def route_current(self) -> int:
        return self._data.get('route_current', 0)
    
    @route_current.setter
    def route_current(self, value: int) -> None:
        self._data['route_current'] = value
    
    @property
    def route_total(self) -> int:
        return self._data.get('route_total', 1)
    
    @route_total.setter
    def route_total(self, value: int) -> None:
        self._data['route_total'] = value
    
    def set_route_message(self, msg: str) -> None:
        """Set the routing status message."""
        self._data['route_message'] = msg[:MSG_SIZE - 1] if msg else ''
    
    def get_route_message(self) -> str:
        return self._data.get('route_message', '')
    
    # ─── Patch (within-cycle) progress ───
    @property
    def patch_current(self) -> int:
        return self._data.get('patch_current', 0)
    
    @patch_current.setter
    def patch_current(self, value: int) -> None:
        self._data['patch_current'] = value
    
    @property
    def patch_total(self) -> int:
        return self._data.get('patch_total', 1)
    
    @patch_total.setter
    def patch_total(self, value: int) -> None:
        self._data['patch_total'] = value
    
    def set_patch_message(self, msg: str) -> None:
        """Set the patch progress message."""
        self._data['patch_message'] = msg[:MSG_SIZE - 1] if msg else ''
    
    def get_patch_message(self) -> str:
        return self._data.get('patch_message', '')
    
    # ─── SAT (inner solver) progress ───
    @property
    def sat_current(self) -> int:
        return self._data.get('sat_current', 0)
    
    @sat_current.setter
    def sat_current(self, value: int) -> None:
        self._data['sat_current'] = value
    
    @property
    def sat_total(self) -> int:
        return self._data.get('sat_total', 1)
    
    @sat_total.setter
    def sat_total(self, value: int) -> None:
        self._data['sat_total'] = value
    
    def set_sat_message(self, msg: str) -> None:
        """Set the SAT solver status message."""
        self._data['sat_message'] = msg[:MSG_SIZE - 1] if msg else ''
    
    def get_sat_message(self) -> str:
        return self._data.get('sat_message', '')
    
    # ─── Legacy aliases (for backward compatibility) ───
    @property
    def current(self) -> int:
        """Legacy alias for sat_current."""
        return self.sat_current
    
    @current.setter
    def current(self, value: int) -> None:
        self.sat_current = value
    
    @property
    def total(self) -> int:
        """Legacy alias for sat_total."""
        return self.sat_total
    
    @total.setter
    def total(self, value: int) -> None:
        self.sat_total = value
    
    # ─── Slot state ───
    @property
    def status(self) -> int:
        return self._data.get('status', STATUS_IDLE)
    
    @status.setter
    def status(self, value: int) -> None:
        self._data['status'] = value
    
    @property
    def config_id(self) -> int:
        return self._data.get('config_id', -1)
    
    @config_id.setter
    def config_id(self, value: int) -> None:
        self._data['config_id'] = value
    
    def reset(self, config_id: int = -1) -> None:
        """Reset slot to initial state for a new config."""
        # Reset routing progress
        self._data['route_current'] = 0
        self._data['route_total'] = 1
        self._data['route_message'] = ''
        # Reset patch progress
        self._data['patch_current'] = 0
        self._data['patch_total'] = 1
        self._data['patch_message'] = ''
        # Reset SAT progress
        self._data['sat_current'] = 0
        self._data['sat_total'] = 1
        self._data['sat_message'] = ''
        # Update state
        self._data['status'] = STATUS_RUNNING if config_id >= 0 else STATUS_IDLE
        self._data['config_id'] = config_id
    
    def set_message(self, msg: str) -> None:
        """Legacy alias for set_sat_message."""
        self.set_sat_message(msg)
    
    def get_message(self) -> str:
        """Legacy alias for get_sat_message."""
        return self.get_sat_message()


@dataclass  
class SharedProgressState:
    """Collection of progress slots for all workers."""
    slots: List[SharedProgressSlot]
    n_workers: int
    manager: Any = None  # Keep reference to manager to prevent GC
    
    @classmethod
    def create(cls, n_workers: int, manager: mp.managers.SyncManager) -> "SharedProgressState":
        """Create shared state for n_workers using the given Manager."""
        return cls(
            slots=[SharedProgressSlot.create(manager) for _ in range(n_workers)],
            n_workers=n_workers,
            manager=manager,
        )
    
    def get_slot(self, worker_id: int) -> SharedProgressSlot:
        """Get the slot for a specific worker."""
        if 0 <= worker_id < self.n_workers:
            return self.slots[worker_id]
        raise IndexError(f"Worker ID {worker_id} out of range [0, {self.n_workers})")


# ═══════════════════════════════════════════════════════════════════════════════
# Progress Callback Factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_shared_progress_callback(
    slot: SharedProgressSlot,
) -> Tuple[Callable, Callable[[], None]]:
    """Create a progress callback that writes to shared memory.
    
    This callback can be used in child processes spawned via 'spawn'
    because it uses Manager proxy objects that ARE picklable.
    
    The callback routes progress updates to THREE progress bars:
    - **Route**: MS rounds progress (route_current/route_total)
    - **Patch**: patch-within-cycle progress (patch_current/patch_total)
    - **SAT**: SAT config solving progress (sat_current/sat_total)
    
    Parameters
    ----------
    slot : SharedProgressSlot
        The shared memory slot to write progress updates to.
    
    Returns
    -------
    tuple of (callback, close)
        callback: Function to call with RoutingProgress updates
        close: Function to call when routing is complete
    """
    # Import here to avoid circular dependency
    from ..compiler.routing_config import (
        _INNER_STAGES,
        _PATCH_STAGES,
        STAGE_ROUTING,
        STAGE_APPLYING,
        STAGE_COMPLETE,
        STAGE_RECONFIG_PROGRESS,
    )
    
    # Stages that update the route (MS round) progress bar
    _ROUTE_STAGES = frozenset({
        STAGE_ROUTING,
        STAGE_APPLYING,
        STAGE_COMPLETE,
        STAGE_RECONFIG_PROGRESS,
    })
    
    def _callback(p) -> None:
        """Update shared memory with progress.
        
        Routes updates to the appropriate progress bar based on stage:
        - Route stages → route_current/route_total (MS rounds + reconfigs)
        - Patch stages → patch_current/patch_total (patches in cycle)
        - Inner stages → sat_current/sat_total (SAT configs)
        """
        stage = p.stage
        
        if stage in _ROUTE_STAGES:
            # Update route (MS round + reconfig) progress bar
            slot.route_current = p.current
            slot.route_total = max(p.total, 1)
            if p.message:
                slot.set_route_message(p.message[:60])
                
        elif stage in _PATCH_STAGES:
            # Update patch (within-cycle) progress bar
            slot.patch_current = p.current
            slot.patch_total = max(p.total, 1)
            if p.message:
                slot.set_patch_message(p.message[:60])
                
        elif stage in _INNER_STAGES:
            # Update SAT (inner solver) progress bar
            slot.sat_current = p.current
            slot.sat_total = max(p.total, 1)
            if p.message:
                slot.set_sat_message(p.message[:60])
    
    def _close() -> None:
        """Mark slot as done."""
        slot.status = STATUS_DONE
    
    return _callback, _close


# ═══════════════════════════════════════════════════════════════════════════════
# Single-Config Notebook Widget Progress (for compile_gadget_for_animation etc.)
# ═══════════════════════════════════════════════════════════════════════════════

def make_notebook_widget_progress_callback(
    title: str = "WISE Routing",
) -> Tuple[Any, "Callable[[Any], None]", "Callable[[], None]"]:
    """Create an ipywidgets-based 4-bar progress callback for notebook use.

    Returns a triple ``(container, callback, close)``:

    - **container**: an ``ipywidgets.VBox`` to ``display()`` in the cell.
    - **callback**: a :pyclass:`RoutingProgress` callback that updates the
      bars directly (no multiprocessing needed).
    - **close**: marks all bars as complete.

    The widget shows a **phase label** at the top describing the current
    work (e.g. "SAT solving EC phase", "Replaying cached Gadget phase")
    and four hierarchical progress bars:

      - **Steps** (blue ``#3498db``):   MS rounds / reconfig steps
      - **Blocks** (teal ``#1abc9c``):  per-block progress when block_level_slicing=True
      - **Patch** (orange ``#e67e22``): patches within tiling cycle
      - **SAT** (green ``#2ecc71``):    SAT configs within patch
    """
    # Import here to avoid circular dependency
    from ..compiler.routing_config import (
        _INNER_STAGES,
        _PATCH_STAGES,
        _BLOCK_STAGES,
        STAGE_ROUTING,
        STAGE_APPLYING,
        STAGE_COMPLETE,
        STAGE_RECONFIG_PROGRESS,
        STAGE_PHASE,
        STAGE_BLOCK,
    )
    _ROUTE_STAGES = frozenset({STAGE_ROUTING, STAGE_APPLYING, STAGE_COMPLETE, STAGE_RECONFIG_PROGRESS})

    import ipywidgets as widgets

    # ── Header + phase label ──────────────────────────────────────
    header = widgets.HTML(
        value=f"<b style='font-size: 13px;'>{title}</b>",
        layout=widgets.Layout(margin='0 0 2px 0'),
    )
    phase_label = widgets.HTML(
        value=(
            "<span style='font-size: 11px; color: #7f8c8d; "
            "font-style: italic;'>Initialising…</span>"
        ),
        layout=widgets.Layout(margin='0 0 4px 0'),
    )

    # ── Steps bar (blue) ──────────────────────────────────────────
    route_bar = widgets.IntProgress(
        value=0, min=0, max=1,
        bar_style='',
        style={'bar_color': '#3498db'},
        layout=widgets.Layout(width='60%'),
    )
    route_label = widgets.HTML(
        value="<span style='color:#3498db; font-size:11px;'>Steps: 0/1</span>",
        layout=widgets.Layout(width='180px', min_width='150px'),
    )
    route_row = widgets.HBox([
        widgets.HTML(
            "<b style='width:80px; display:inline-block; font-size:12px;'>Steps</b>",
            layout=widgets.Layout(width='80px'),
        ),
        route_bar,
        route_label,
    ], layout=widgets.Layout(align_items='center'))

    # ── Blocks bar (teal) — hidden when there's no block-level slicing ─
    block_bar = widgets.IntProgress(
        value=0, min=0, max=1,
        bar_style='',
        style={'bar_color': '#1abc9c'},
        layout=widgets.Layout(width='60%'),
    )
    block_label = widgets.HTML(
        value="<span style='color:#1abc9c; font-size:11px;'>Blocks: –</span>",
        layout=widgets.Layout(width='180px', min_width='150px'),
    )
    block_row = widgets.HBox([
        widgets.HTML(
            "<b style='width:80px; display:inline-block; font-size:12px;'>Blocks</b>",
            layout=widgets.Layout(width='80px'),
        ),
        block_bar,
        block_label,
    ], layout=widgets.Layout(align_items='center', display='none'))

    # ── Patch bar (orange) ────────────────────────────────────────
    patch_bar = widgets.IntProgress(
        value=0, min=0, max=1,
        bar_style='',
        style={'bar_color': '#e67e22'},
        layout=widgets.Layout(width='60%'),
    )
    patch_label = widgets.HTML(
        value="<span style='color:#e67e22; font-size:11px;'>Patches: 0/1</span>",
        layout=widgets.Layout(width='180px', min_width='150px'),
    )
    patch_row = widgets.HBox([
        widgets.HTML(
            "<b style='width:80px; display:inline-block; font-size:12px;'>Patch</b>",
            layout=widgets.Layout(width='80px'),
        ),
        patch_bar,
        patch_label,
    ], layout=widgets.Layout(align_items='center'))

    # ── SAT bar (green) ──────────────────────────────────────────
    sat_bar = widgets.IntProgress(
        value=0, min=0, max=1,
        bar_style='',
        style={'bar_color': '#2ecc71'},
        layout=widgets.Layout(width='60%'),
    )
    sat_label = widgets.HTML(
        value="<span style='color:#2ecc71; font-size:11px;'>SAT: 0/1</span>",
        layout=widgets.Layout(width='180px', min_width='150px'),
    )
    sat_row = widgets.HBox([
        widgets.HTML(
            "<b style='width:80px; display:inline-block; font-size:12px;'>SAT</b>",
            layout=widgets.Layout(width='80px'),
        ),
        sat_bar,
        sat_label,
    ], layout=widgets.Layout(align_items='center'))

    # ── Container ─────────────────────────────────────────────────
    container = widgets.VBox(
        [header, phase_label, route_row, block_row, patch_row, sat_row],
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='8px',
            margin='4px 0',
            width='100%',
        ),
    )

    # ── Callback ──────────────────────────────────────────────────
    # Hierarchical reset model:
    #   Phase (informational): label describing current QEC phase.
    #   Steps (outermost):  shows completed MS rounds — only moves forward.
    #   Blocks (optional):  shows per-block progress within a step —
    #                       resets each time a new step starts.
    #   Patch (middle):     shows patches in current tiling cycle — resets
    #                       each time a new block (or step) starts.
    #   SAT (innermost):    shows configs in current SAT solve — resets
    #                       each time a new patch starts solving.
    #
    # THROTTLING: to avoid overwhelming the IOPub channel with widget
    # comm messages (which causes "IOStream.flush timed out" and hangs),
    # we accumulate state changes internally and only push to widgets
    # at most once every _FLUSH_INTERVAL seconds.
    import time as _time_mod

    _route_hwm = 0
    _FLUSH_INTERVAL = 0.25          # seconds between widget pushes
    _last_push: float = 0.0         # monotonic ts of last widget push
    _blocks_visible = False         # whether the block bar has been shown

    # Shadow state — updated on every callback, pushed to widgets on flush
    _s_phase_text = (
        "<span style='font-size: 11px; color: #7f8c8d; "
        "font-style: italic;'>Initialising…</span>"
    )
    _s_route_val = 0
    _s_route_max = 1
    _s_route_text = "<span style='color:#3498db; font-size:11px;'>Steps: 0/1</span>"
    _s_block_val = 0
    _s_block_max = 1
    _s_block_style = ''
    _s_block_text = "<span style='color:#1abc9c; font-size:11px;'>Blocks: –</span>"
    _s_patch_val = 0
    _s_patch_max = 1
    _s_patch_style = ''
    _s_patch_text = "<span style='color:#e67e22; font-size:11px;'>Patches: 0/1</span>"
    _s_sat_val = 0
    _s_sat_max = 1
    _s_sat_style = ''
    _s_sat_text = "<span style='color:#2ecc71; font-size:11px;'>SAT: 0/1</span>"

    def _push_to_widgets(force: bool = False) -> None:
        """Push accumulated shadow state → widgets (throttled)."""
        nonlocal _last_push, _blocks_visible
        now = _time_mod.monotonic()
        if not force and (now - _last_push) < _FLUSH_INTERVAL:
            return
        _last_push = now
        # Batch all widget mutations together
        phase_label.value = _s_phase_text
        route_bar.max = _s_route_max
        route_bar.value = _s_route_val
        route_label.value = _s_route_text
        block_bar.max = _s_block_max
        block_bar.value = _s_block_val
        block_bar.bar_style = _s_block_style
        block_label.value = _s_block_text
        if _blocks_visible:
            block_row.layout.display = ''
        patch_bar.max = _s_patch_max
        patch_bar.value = _s_patch_val
        patch_bar.bar_style = _s_patch_style
        patch_label.value = _s_patch_text
        sat_bar.max = _s_sat_max
        sat_bar.value = _s_sat_val
        sat_bar.bar_style = _s_sat_style
        sat_label.value = _s_sat_text

    def _callback(p) -> None:
        nonlocal _route_hwm, _blocks_visible
        nonlocal _s_phase_text
        nonlocal _s_route_val, _s_route_max, _s_route_text
        nonlocal _s_block_val, _s_block_max, _s_block_style, _s_block_text
        nonlocal _s_patch_val, _s_patch_max, _s_patch_style, _s_patch_text
        nonlocal _s_sat_val, _s_sat_max, _s_sat_style, _s_sat_text
        stage = p.stage
        force_push = False

        # ── Phase label (informational) ───────────────────────────
        if stage == STAGE_PHASE:
            # p.message carries the human-readable phase description
            # p.current / p.total carry phase index / total phases
            _phase_msg = p.message or "Working…"
            _phase_idx = p.current
            _phase_total = max(p.total, 1)
            _s_phase_text = (
                f"<span style='font-size: 11px; color: #2c3e50;'>"
                f"<b>Phase {_phase_idx + 1}/{_phase_total}:</b> "
                f"{_phase_msg}</span>"
            )
            force_push = True

        # ── Block progress ────────────────────────────────────────
        elif stage == STAGE_BLOCK:
            _blocks_visible = True
            total = max(p.total, 1)
            cur = min(p.current, total)
            _s_block_max = total
            _s_block_val = cur
            _block_name = p.message or ""
            _s_block_style = 'success' if cur >= total else ''
            _s_block_text = (
                f"<span style='color:#1abc9c; font-size:11px;'>"
                f"Blocks: {cur}/{total}"
                f"{(' (' + _block_name + ')') if _block_name else ''}"
                f"</span>"
            )
            # New block → reset sub-bars
            _s_patch_val = 0
            _s_patch_max = 1
            _s_patch_style = ''
            _s_patch_text = (
                "<span style='color:#e67e22; font-size:11px;'>"
                "Patches: 0/?</span>"
            )
            _s_sat_val = 0
            _s_sat_max = 1
            _s_sat_style = ''
            _s_sat_text = (
                "<span style='color:#2ecc71; font-size:11px;'>"
                "SAT: waiting</span>"
            )
            force_push = True

        # ── Steps (route) progress ────────────────────────────────
        elif stage in _ROUTE_STAGES:
            total = max(p.total, 1)
            cur = min(p.current, total)
            # STAGE_APPLYING signals the start of the execution phase
            # (applying precomputed routing steps).  Reset the HWM so
            # execution progress can show on the bar instead of being
            # blocked by the planning-phase high-water mark.
            if stage == STAGE_APPLYING:
                _route_hwm = -1          # reset HWM for execution phase
            # Only allow forward progress (HWM blocks per-block re-emissions)
            if total > _s_route_max or cur >= _route_hwm:
                old_route = _s_route_val
                _route_hwm = cur
                _s_route_max = total
                _s_route_val = cur
                # Switch label between planning / applying / reconfig
                # based on stage so users can see what is happening.
                if stage == STAGE_RECONFIG_PROGRESS:
                    _s_route_text = (
                        f"<span style='color:#9b59b6; font-size:11px;'>"
                        f"Reconfig: {cur}/{total}</span>"
                    )
                elif stage == STAGE_APPLYING:
                    _s_route_text = (
                        f"<span style='color:#8e44ad; font-size:11px;'>"
                        f"Applying: {cur}/{total}</span>"
                    )
                else:
                    _s_route_text = (
                        f"<span style='color:#3498db; font-size:11px;'>"
                        f"Steps: {cur}/{total}</span>"
                    )
                # Only reset sub-bars when Route truly advances
                if cur > old_route:
                    # Reset blocks bar
                    _s_block_val = 0
                    _s_block_max = 1
                    _s_block_style = ''
                    _s_block_text = (
                        "<span style='color:#1abc9c; font-size:11px;'>"
                        "Blocks: –</span>"
                    )
                    _s_patch_val = 0
                    _s_patch_max = 1
                    _s_patch_style = ''
                    _s_patch_text = (
                        "<span style='color:#e67e22; font-size:11px;'>"
                        "Patches: 0/?</span>"
                    )
                    _s_sat_val = 0
                    _s_sat_max = 1
                    _s_sat_style = ''
                    _s_sat_text = (
                        "<span style='color:#2ecc71; font-size:11px;'>"
                        "SAT: waiting</span>"
                    )
                    force_push = True  # Route advance → always push

        elif stage in _PATCH_STAGES:
            total = max(p.total, 1)
            cur = min(p.current, total)
            _s_patch_max = total
            _s_patch_val = cur
            _s_patch_style = 'success' if cur >= total else ''
            _s_patch_text = (
                f"<span style='color:#e67e22; font-size:11px;'>"
                f"Patches: {cur}/{total}</span>"
            )
            # New patch started → reset SAT bar
            if cur == 0 or (cur > 0 and _s_sat_val >= _s_sat_max > 1):
                _s_sat_val = 0
                _s_sat_max = 1
                _s_sat_style = ''
                _s_sat_text = (
                    "<span style='color:#2ecc71; font-size:11px;'>"
                    "SAT: waiting</span>"
                )
            force_push = True  # Patch updates should be visible immediately

        elif stage in _INNER_STAGES:
            total = max(p.total, 1)
            cur = min(p.current, total)
            _s_sat_max = total
            _s_sat_val = cur
            _s_sat_style = 'success' if cur >= total else ''
            _s_sat_text = (
                f"<span style='color:#2ecc71; font-size:11px;'>"
                f"SAT: {cur}/{total}</span>"
            )
            force_push = True  # SAT updates should be visible immediately

        _push_to_widgets(force=force_push)

    # ── Close ─────────────────────────────────────────────────────
    def _close() -> None:
        nonlocal _s_route_val, _s_block_val, _s_patch_val, _s_sat_val
        nonlocal _s_block_style, _s_patch_style, _s_sat_style
        _s_route_val = _s_route_max
        _s_block_val = _s_block_max
        _s_patch_val = _s_patch_max
        _s_sat_val = _s_sat_max
        _s_block_style = 'success'
        _s_patch_style = 'success'
        _s_sat_style = 'success'
        _push_to_widgets(force=True)
        route_bar.bar_style = 'success'
        block_bar.bar_style = 'success'
        patch_bar.bar_style = 'success'
        sat_bar.bar_style = 'success'
        header.value = (
            f"<b style='font-size: 13px; color: #27ae60;'>"
            f"✓ {title} — complete</b>"
        )
        phase_label.value = ""

    return container, _callback, _close


# ═══════════════════════════════════════════════════════════════════════════════
# ipywidgets Progress Table (Notebook Mode)
# ═══════════════════════════════════════════════════════════════════════════════

def _in_notebook() -> bool:
    """Check if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        return shell_name in ('ZMQInteractiveShell', 'Shell')
    except Exception:
        return False


class ProgressTableWidget:
    """ipywidgets-based progress table for parallel WISE configs.
    
    Displays a dynamic table with one row per active worker, showing:
    - Config name (e.g., "la=1, (4,3,0)")
    - Progress bar
    - Status message
    
    Usage:
        state = SharedProgressState.create(n_workers=4)
        widget = ProgressTableWidget(state, config_labels)
        display(widget.container)
        
        # In polling loop:
        widget.refresh()
        
        # When done:
        widget.close()
    """
    
    def __init__(
        self,
        shared_state: SharedProgressState,
        config_labels: List[str],
        outer_desc: str = "WISE Configs",
    ):
        """Initialize the progress table.
        
        Parameters
        ----------
        shared_state : SharedProgressState
            Shared memory state for all workers.
        config_labels : List[str]
            Human-readable labels for each config (indexed by config_id).
        outer_desc : str
            Description for the outer (config-level) progress bar.
        """
        self.shared_state = shared_state
        self.config_labels = config_labels
        self.outer_desc = outer_desc
        self.n_workers = shared_state.n_workers
        self.n_configs = len(config_labels)
        self._closed = False
        self._completed_count = 0
        
        # Try to import ipywidgets
        try:
            import ipywidgets as widgets
            from IPython.display import display
            self._widgets = widgets
            self._display = display
            self._use_widgets = _in_notebook()
        except ImportError:
            self._use_widgets = False
        
        if self._use_widgets:
            self._init_widgets()
        else:
            self._init_tqdm()
    
    def _init_widgets(self):
        """Initialize ipywidgets UI with triple progress bars per worker."""
        widgets = self._widgets
        
        # Outer progress bar (config-level)
        self.outer_progress = widgets.IntProgress(
            value=0,
            min=0,
            max=self.n_configs,
            description=self.outer_desc,
            bar_style='info',
            style={'bar_color': '#3498db', 'description_width': '100px'},
            layout=widgets.Layout(width='100%'),
        )
        self.outer_label = widgets.HTML(
            value=f"<b>0/{self.n_configs}</b> configs",
            layout=widgets.Layout(width='150px'),
        )
        self.outer_box = widgets.HBox([
            self.outer_progress,
            self.outer_label,
        ], layout=widgets.Layout(margin='0 0 10px 0'))
        
        # Worker rows (one per worker slot) - 3 bars each
        self.worker_rows = []
        self.worker_labels = []
        self.worker_route_bars = []    # MS rounds
        self.worker_route_labels = []
        self.worker_patch_bars = []    # Patches in cycle
        self.worker_patch_labels = []
        self.worker_sat_bars = []      # SAT configs
        self.worker_sat_labels = []
        
        for i in range(self.n_workers):
            # Worker label (config name)
            label = widgets.HTML(
                value=f"<span style='color: #888;'>Worker {i}: idle</span>",
                layout=widgets.Layout(width='160px', min_width='130px'),
            )
            
            # Route progress bar (blue - MS rounds)
            route_bar = widgets.IntProgress(
                value=0, min=0, max=100,
                bar_style='',
                style={'bar_color': '#3498db'},
                layout=widgets.Layout(width='22%', visibility='hidden'),
            )
            route_label = widgets.HTML(
                value="",
                layout=widgets.Layout(width='80px', min_width='70px'),
            )
            
            # Patch progress bar (orange - patches in cycle)
            patch_bar = widgets.IntProgress(
                value=0, min=0, max=100,
                bar_style='',
                style={'bar_color': '#e67e22'},
                layout=widgets.Layout(width='22%', visibility='hidden'),
            )
            patch_label = widgets.HTML(
                value="",
                layout=widgets.Layout(width='80px', min_width='70px'),
            )
            
            # SAT progress bar (green - SAT configs)
            sat_bar = widgets.IntProgress(
                value=0, min=0, max=100,
                bar_style='',
                style={'bar_color': '#2ecc71'},
                layout=widgets.Layout(width='22%', visibility='hidden'),
            )
            sat_label = widgets.HTML(
                value="",
                layout=widgets.Layout(width='80px', min_width='70px'),
            )
            
            # Row layout: label | route_bar route_label | patch_bar patch_label | sat_bar sat_label
            row = widgets.HBox([
                label,
                route_bar, route_label,
                patch_bar, patch_label,
                sat_bar, sat_label,
            ], layout=widgets.Layout(align_items='center'))
            
            self.worker_labels.append(label)
            self.worker_route_bars.append(route_bar)
            self.worker_route_labels.append(route_label)
            self.worker_patch_bars.append(patch_bar)
            self.worker_patch_labels.append(patch_label)
            self.worker_sat_bars.append(sat_bar)
            self.worker_sat_labels.append(sat_label)
            self.worker_rows.append(row)
        
        # Container
        self.container = widgets.VBox([
            self.outer_box,
            widgets.VBox(self.worker_rows),
        ])
    
    def _init_tqdm(self):
        """Initialize tqdm fallback for terminal mode."""
        try:
            from tqdm.auto import tqdm
            self._tqdm = tqdm
            self.outer_bar = tqdm(
                total=self.n_configs,
                desc=self.outer_desc,
                unit="cfg",
                position=0,
                leave=True,
            )
            # Create nested bars for workers
            self.worker_tqdm_bars = {}
        except ImportError:
            self._tqdm = None
            self.outer_bar = None
            self.worker_tqdm_bars = {}
    
    def refresh(self) -> None:
        """Refresh the display from shared memory state."""
        if self._closed:
            return
        
        if self._use_widgets:
            self._refresh_widgets()
        else:
            self._refresh_tqdm()
    
    def _refresh_widgets(self):
        """Refresh ipywidgets display with triple progress bars per worker."""
        for i in range(self.n_workers):
            slot = self.shared_state.slots[i]
            status_code = slot.status
            config_id = slot.config_id
            
            if status_code == STATUS_IDLE:
                self.worker_labels[i].value = f"<span style='color: #888;'>Worker {i}: idle</span>"
                self.worker_route_bars[i].layout.visibility = 'hidden'
                self.worker_route_labels[i].value = ""
                self.worker_patch_bars[i].layout.visibility = 'hidden'
                self.worker_patch_labels[i].value = ""
                self.worker_sat_bars[i].layout.visibility = 'hidden'
                self.worker_sat_labels[i].value = ""
                
            elif status_code == STATUS_RUNNING:
                cfg_label = self.config_labels[config_id] if 0 <= config_id < len(self.config_labels) else f"cfg {config_id}"
                self.worker_labels[i].value = f"<b>W{i}:</b> {cfg_label}"
                
                # ─── Update route (MS round) progress bar ───
                route_current = slot.route_current
                route_total = max(slot.route_total, 1)
                
                self.worker_route_bars[i].max = route_total
                self.worker_route_bars[i].value = min(route_current, route_total)
                self.worker_route_bars[i].layout.visibility = 'visible'
                self.worker_route_labels[i].value = (
                    f"<span style='color: #3498db; font-size: 11px;'>"
                    f"Rte {route_current}/{route_total}</span>"
                )
                
                # ─── Update patch progress bar ───
                patch_current = slot.patch_current
                patch_total = max(slot.patch_total, 1)
                
                self.worker_patch_bars[i].max = patch_total
                self.worker_patch_bars[i].value = min(patch_current, patch_total)
                self.worker_patch_bars[i].layout.visibility = 'visible'
                self.worker_patch_labels[i].value = (
                    f"<span style='color: #e67e22; font-size: 11px;'>"
                    f"Pat {patch_current}/{patch_total}</span>"
                )
                
                # ─── Update SAT progress bar ───
                sat_current = slot.sat_current
                sat_total = max(slot.sat_total, 1)
                
                self.worker_sat_bars[i].max = sat_total
                self.worker_sat_bars[i].value = min(sat_current, sat_total)
                self.worker_sat_bars[i].layout.visibility = 'visible'
                self.worker_sat_labels[i].value = (
                    f"<span style='color: #27ae60; font-size: 11px;'>"
                    f"SAT {sat_current}/{sat_total}</span>"
                )
                
            elif status_code == STATUS_DONE:
                cfg_label = self.config_labels[config_id] if 0 <= config_id < len(self.config_labels) else f"cfg {config_id}"
                self.worker_labels[i].value = f"<span style='color: #27ae60;'>W{i}: \u2713 {cfg_label}</span>"
                
                # Mark all three bars as complete
                self.worker_route_bars[i].value = self.worker_route_bars[i].max
                self.worker_route_bars[i].bar_style = 'success'
                self.worker_route_labels[i].value = "<span style='color: #27ae60; font-size: 11px;'>done</span>"
                
                self.worker_patch_bars[i].value = self.worker_patch_bars[i].max
                self.worker_patch_bars[i].bar_style = 'success'
                self.worker_patch_labels[i].value = "<span style='color: #27ae60; font-size: 11px;'>done</span>"
                
                self.worker_sat_bars[i].value = self.worker_sat_bars[i].max
                self.worker_sat_bars[i].bar_style = 'success'
                self.worker_sat_labels[i].value = "<span style='color: #27ae60; font-size: 11px;'>done</span>"
                
            elif status_code == STATUS_ERROR:
                cfg_label = self.config_labels[config_id] if 0 <= config_id < len(self.config_labels) else f"cfg {config_id}"
                self.worker_labels[i].value = f"<span style='color: #e74c3c;'>W{i}: \u2717 {cfg_label}</span>"
                self.worker_route_bars[i].bar_style = 'danger'
                self.worker_patch_bars[i].bar_style = 'danger'
                self.worker_sat_bars[i].bar_style = 'danger'
                err_msg = slot.get_sat_message() or "error"
                self.worker_sat_labels[i].value = f"<span style='color: #e74c3c; font-size: 11px;'>{err_msg[:20]}</span>"
    
    def _refresh_tqdm(self):
        """Refresh tqdm display."""
        if self._tqdm is None:
            return
        
        # Just update outer bar - worker bars are too noisy in terminal
        pass
    
    def update_completed(self, count: int) -> None:
        """Update the count of completed configs."""
        self._completed_count = count
        if self._use_widgets:
            self.outer_progress.value = count
            self.outer_label.value = f"<b>{count}/{self.n_configs}</b> configs"
        elif self.outer_bar is not None:
            self.outer_bar.n = count
            self.outer_bar.refresh()
    
    def close(self) -> None:
        """Close the progress display."""
        if self._closed:
            return
        self._closed = True
        
        if self._use_widgets:
            # Mark all bars as complete
            self.outer_progress.bar_style = 'success'
            for bar in self.worker_route_bars:
                bar.bar_style = 'success'
            for bar in self.worker_patch_bars:
                bar.bar_style = 'success'
            for bar in self.worker_sat_bars:
                bar.bar_style = 'success'
        elif self.outer_bar is not None:
            self.outer_bar.close()
            for bar in self.worker_tqdm_bars.values():
                bar.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Polling Thread for Widget Updates
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressPoller:
    """Background thread that polls shared state and updates the widget.
    
    This runs in the main process and periodically reads from shared memory
    to update the ipywidgets display.
    """
    
    def __init__(
        self,
        widget: ProgressTableWidget,
        poll_interval: float = 0.1,
    ):
        self.widget = widget
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the polling thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def _poll_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            try:
                self.widget.refresh()
            except Exception:
                pass  # Ignore errors during refresh
            time.sleep(self.poll_interval)
