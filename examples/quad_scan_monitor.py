"""
quad_scan_monitor.py
--------------------
Live beam monitor – quadrupole scan with streaming plots.

Run with:
    marimo run quad_scan_monitor.py           # app mode (clean UI)
    marimo edit quad_scan_monitor.py          # notebook edit mode

The data source is swappable:
  - MockImageSource  – synthetic data, no hardware/lattice needed
  - ModelImageSource – wraps the SLAC StagedModel (requires LCLS_LATTICE env var)
"""

import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full", app_title="Live Beam Monitor")


# ── Cell 0: imports & configuration ─────────────────────────────────────────
@app.cell
def imports():
    import asyncio
    import os
    import sys
    import warnings
    from collections import deque
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend – required for marimo
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    import numpy as np
    import marimo as mo

    warnings.filterwarnings("ignore")

    # ── user-facing configuration ────────────────────────────────────────────
    LCLS_LATTICE_PATH = "/Users/smiskov/SLAC/lcls-lattice"
    SCAN_PV = "QUAD:IN20:525:BCTRL"
    IMAGE_PV = "OTRS:IN20:711:Image:ArrayData"  # match test_example.py
    XRMS_PV = "OTRS:IN20:571:XRMS"
    YRMS_PV = "OTRS:IN20:571:YRMS"
    MAX_HISTORY = 200

    # Directory containing this notebook – used for importing beam_monitor.py
    NB_DIR = str(Path(__file__).parent.resolve())

    return (
        asyncio,
        os,
        sys,
        warnings,
        deque,
        Path,
        matplotlib,
        plt,
        gridspec,
        ticker,
        np,
        mo,
        LCLS_LATTICE_PATH,
        SCAN_PV,
        IMAGE_PV,
        XRMS_PV,
        YRMS_PV,
        MAX_HISTORY,
        NB_DIR,
    )


# ── Cell 0b: page header (always visible) ────────────────────────────────────
@app.cell
def header(mo):
    header_md = mo.md(
        "# Live Beam Monitor\nConfigure the scan below, then press **▶ Run Scan**."
    )
    header_md
    return (header_md,)


# ── Cell 1: UI controls ──────────────────────────────────────────────────────
@app.cell
def controls(mo):
    use_mock_toggle = mo.ui.switch(
        value=True, label="Use mock source (no lattice needed)"
    )
    mock_wait_slider = mo.ui.slider(
        start=0.1, stop=3.0, step=0.1, value=1.0, label="Mock wait per step (s)"
    )
    scan_min_slider = mo.ui.slider(
        start=-15.0, stop=0.0, step=0.5, value=-10.0, label="Quad min (kG)"
    )
    scan_max_slider = mo.ui.slider(
        start=0.0, stop=15.0, step=0.5, value=10.0, label="Quad max (kG)"
    )
    scan_steps_slider = mo.ui.slider(start=3, stop=20, step=1, value=5, label="Steps")
    run_button = mo.ui.run_button(label="▶  Run Scan")
    stop_button = mo.ui.button(label="⏹  Stop", kind="danger")

    controls_ui = mo.vstack(
        [
            mo.md("## Scan Configuration"),
            mo.hstack([use_mock_toggle, mock_wait_slider], justify="start", gap="2rem"),
            mo.hstack(
                [scan_min_slider, scan_max_slider, scan_steps_slider], gap="2rem"
            ),
            mo.hstack([run_button, stop_button], gap="1rem"),
        ]
    )
    controls_ui

    return (
        use_mock_toggle,
        mock_wait_slider,
        scan_min_slider,
        scan_max_slider,
        scan_steps_slider,
        run_button,
        stop_button,
        controls_ui,
    )


# ── Cell 2: reactive state ───────────────────────────────────────────────────
@app.cell
def state(mo, NB_DIR, sys):
    if NB_DIR not in sys.path:
        sys.path.insert(0, NB_DIR)
    from beam_monitor import BeamFrame

    status_text, set_status = mo.state("Idle — press **▶ Run Scan** to start.")
    scan_running, set_scan_running = mo.state(False)
    run_token, set_run_token = mo.state(0)
    real_source, set_real_source = mo.state(None)

    return (
        BeamFrame,
        status_text,
        set_status,
        scan_running,
        set_scan_running,
        run_token,
        set_run_token,
        real_source,
        set_real_source,
    )


# ── Cell 3: persistent dashboard setup ───────────────────────────────────────
@app.cell
def dashboard(plt, gridspec, ticker, np, SCAN_PV):
    DARK = "#0d1117"
    PANEL = "#161b22"
    GRID = "#30363d"
    TXT = "#c9d1d9"
    BLUE = "#58a6ff"
    CORAL = "#f78166"
    GREEN = "#3fb950"
    PURPLE = "#d2a8ff"

    def _style_ax(ax, title=""):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        if title:
            ax.set_title(title, color=TXT, fontsize=9, pad=6)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)

    fig = plt.figure(figsize=(13, 9), facecolor=DARK)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.38,
        wspace=0.32,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.09,
    )

    # image panel
    ax_img = fig.add_subplot(gs[0, 0])
    _style_ax(ax_img, "OTR Beam Image")
    image_artist = ax_img.imshow(
        np.zeros((2, 2)),
        cmap="inferno",
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        visible=False,
    )
    colorbar = fig.colorbar(image_artist, ax=ax_img, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=TXT, labelsize=7)
    colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
    colorbar.formatter.set_scientific(True)
    colorbar.formatter.set_powerlimits((0, 0))
    colorbar.update_ticks()
    image_placeholder = ax_img.text(
        0.5,
        0.5,
        "Waiting for first shot…",
        ha="center",
        va="center",
        color=TXT,
        transform=ax_img.transAxes,
        fontsize=11,
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_xlabel("", fontsize=8)

    # scatter panel
    ax_ps = fig.add_subplot(gs[0, 1])
    _style_ax(ax_ps, "Beam Phase-Space  x – y")
    scatter_artist = ax_ps.scatter(
        [], [], s=0.8, alpha=0.35, color=BLUE, rasterized=True
    )
    scatter_placeholder = ax_ps.text(
        0.5,
        0.5,
        "Waiting for first shot…",
        ha="center",
        va="center",
        color=TXT,
        transform=ax_ps.transAxes,
        fontsize=11,
    )
    ax_ps.set_xlabel("x  (µm)", fontsize=8)
    ax_ps.set_ylabel("y  (µm)", fontsize=8)

    # timeseries panel
    ax_ts = fig.add_subplot(gs[1, :])
    _style_ax(ax_ts, "Scalar Diagnostics vs Quad Setting")
    line_x = ax_ts.plot(
        [], [], color=BLUE, lw=1.8, marker="o", ms=5, label="σ_x  (µm)"
    )[0]
    line_y = ax_ts.plot(
        [], [], color=CORAL, lw=1.8, marker="s", ms=5, label="σ_y  (µm)"
    )[0]
    line_z = ax_ts.plot(
        [], [], color="#e3b341", lw=1.8, marker="D", ms=4, label="σ_z  (µm)"
    )[0]
    ax_ts.set_xlabel(f"{SCAN_PV}  (kG)", fontsize=8)
    ax_ts.set_ylabel("RMS beam size  (µm)", fontsize=8)
    vline = ax_ts.axvline(0.0, color="white", lw=1.0, alpha=0.5, linestyle=":")
    vline.set_visible(False)

    ax_em = ax_ts.twinx()
    ax_em.set_facecolor(PANEL)
    ax_em.tick_params(colors=TXT, labelsize=8)
    ax_em.yaxis.label.set_color(TXT)
    for sp in ax_em.spines.values():
        sp.set_color(GRID)
    line_emx = ax_em.plot(
        [],
        [],
        color=GREEN,
        lw=1.8,
        marker="^",
        ms=5,
        linestyle="--",
        label="ε_n,x  (µm·rad)",
    )[0]
    line_emy = ax_em.plot(
        [],
        [],
        color=PURPLE,
        lw=1.8,
        marker="v",
        ms=5,
        linestyle="--",
        label="ε_n,y  (µm·rad)",
    )[0]
    ax_em.set_ylabel("Norm. emittance  (µm·rad)", fontsize=8)
    timeseries_placeholder = ax_ts.text(
        0.5,
        0.5,
        "Scalar time-series will appear here once the scan starts…",
        ha="center",
        va="center",
        color=TXT,
        transform=ax_ts.transAxes,
        fontsize=11,
    )

    l1, lb1 = ax_ts.get_legend_handles_labels()
    l2, lb2 = ax_em.get_legend_handles_labels()
    ax_ts.legend(
        l1 + l2,
        lb1 + lb2,
        loc="upper right",
        fontsize=7,
        facecolor=DARK,
        edgecolor=GRID,
        labelcolor=TXT,
    )

    title_text = fig.suptitle(
        "Live Beam Monitor  ·  idle", color=TXT, fontsize=11, y=0.98
    )

    history_data = {
        "quads": [],
        "xrms": [],
        "yrms": [],
        "sigmaz": [],
        "emx": [],
        "emy": [],
    }

    def _pad_bounds(values, fraction=0.08, minimum=1.0):
        if len(values) == 0:
            return (-1.0, 1.0)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            pad = max(abs(vmin) * fraction, minimum)
            return (vmin - pad, vmax + pad)
        pad = max((vmax - vmin) * fraction, minimum)
        return (vmin - pad, vmax + pad)

    def reset_dashboard():
        history_data["quads"].clear()
        history_data["xrms"].clear()
        history_data["yrms"].clear()
        history_data["sigmaz"].clear()
        history_data["emx"].clear()
        history_data["emy"].clear()

        image_artist.set_visible(False)
        image_placeholder.set_visible(True)
        ax_img.set_xlabel("", fontsize=8)

        scatter_artist.set_offsets(np.empty((0, 2)))
        scatter_placeholder.set_visible(True)

        line_x.set_data([], [])
        line_y.set_data([], [])
        line_z.set_data([], [])
        line_emx.set_data([], [])
        line_emy.set_data([], [])
        vline.set_visible(False)
        timeseries_placeholder.set_visible(True)
        title_text.set_text("Live Beam Monitor  ·  idle")

        ax_ts.set_xlim(-1.0, 1.0)
        ax_ts.set_ylim(-1.0, 1.0)
        ax_em.set_ylim(-1.0, 1.0)
        ax_ps.set_xlim(-1.0, 1.0)
        ax_ps.set_ylim(-1.0, 1.0)

        fig.canvas.draw_idle()
        if hasattr(fig.canvas, "flush_events"):
            fig.canvas.flush_events()

    def update_dashboard(frame):
        if frame.image is not None:
            image_placeholder.set_visible(False)
            image_artist.set_visible(True)
            display_image = -np.asarray(frame.image)  # match test_example.py
            image_artist.set_data(display_image)
            finite = display_image[np.isfinite(display_image)]
            if finite.size:
                _img_min = float(np.min(finite))
                _img_max = float(np.max(finite))
            else:
                _img_min, _img_max = 0.0, 1.0
            if np.isclose(_img_min, _img_max):
                _img_max = _img_min + max(abs(_img_min) * 0.1, 1e-18)
            image_artist.set_clim(_img_min, _img_max)
            colorbar.update_normal(image_artist)
            colorbar.update_ticks()
            ax_img.set_xlabel(f"{SCAN_PV} = {frame.scan_value:.2f} kG", fontsize=8)

        if (
            frame.beam_x is not None
            and frame.beam_y is not None
            and len(frame.beam_x) > 0
        ):
            scatter_placeholder.set_visible(False)
            offsets = np.column_stack((frame.beam_x, frame.beam_y))
            scatter_artist.set_offsets(offsets)
            xpad = max(
                (float(np.max(frame.beam_x)) - float(np.min(frame.beam_x))) * 0.08, 1.0
            )
            ypad = max(
                (float(np.max(frame.beam_y)) - float(np.min(frame.beam_y))) * 0.08, 1.0
            )
            ax_ps.set_xlim(
                float(np.min(frame.beam_x)) - xpad, float(np.max(frame.beam_x)) + xpad
            )
            ax_ps.set_ylim(
                float(np.min(frame.beam_y)) - ypad, float(np.max(frame.beam_y)) + ypad
            )

        history_data["quads"].append(frame.scan_value)
        history_data["xrms"].append(frame.xrms)
        history_data["yrms"].append(frame.yrms)
        history_data["sigmaz"].append(frame.sigma_z * 1e6)
        history_data["emx"].append(frame.norm_emit_x * 1e6)
        history_data["emy"].append(frame.norm_emit_y * 1e6)

        timeseries_placeholder.set_visible(False)
        line_x.set_data(history_data["quads"], history_data["xrms"])
        line_y.set_data(history_data["quads"], history_data["yrms"])
        line_z.set_data(history_data["quads"], history_data["sigmaz"])
        line_emx.set_data(history_data["quads"], history_data["emx"])
        line_emy.set_data(history_data["quads"], history_data["emy"])
        vline.set_xdata([frame.scan_value, frame.scan_value])
        vline.set_visible(True)

        ax_ts.set_xlim(*_pad_bounds(history_data["quads"], minimum=0.5))
        ax_ts.set_ylim(
            *_pad_bounds(
                history_data["xrms"] + history_data["yrms"] + history_data["sigmaz"],
                minimum=5.0,
            )
        )
        ax_em.set_ylim(
            *_pad_bounds(history_data["emx"] + history_data["emy"], minimum=0.05)
        )
        title_text.set_text("Live Beam Monitor")

        fig.canvas.draw_idle()
        if hasattr(fig.canvas, "flush_events"):
            fig.canvas.flush_events()

    return fig, reset_dashboard, update_dashboard


# ── Cell 4: dashboard display ────────────────────────────────────────────────
@app.cell
def dashboard_view(fig, mo):
    dashboard_widget = mo.mpl.interactive(fig)
    dashboard_widget
    return (dashboard_widget,)


# ── Cell 5: scan launcher ────────────────────────────────────────────────────
@app.cell
async def scan_task(
    run_button,
    use_mock_toggle,
    mock_wait_slider,
    scan_min_slider,
    scan_max_slider,
    scan_steps_slider,
    reset_dashboard,
    update_dashboard,
    set_status,
    set_scan_running,
    run_token,
    set_run_token,
    real_source,
    set_real_source,
    np,
    asyncio,
    mo,
    os,
    NB_DIR,
    sys,
    LCLS_LATTICE_PATH,
    SCAN_PV,
    IMAGE_PV,
    XRMS_PV,
    YRMS_PV,
):
    mo.stop(not run_button.value)

    if NB_DIR not in sys.path:
        sys.path.insert(0, NB_DIR)
    from beam_monitor import MockImageSource, ModelImageSource

    _token = run_token() + 1
    set_run_token(_token)

    reset_dashboard()
    set_status("Preparing new scan…")
    set_scan_running(True)

    async def _run_scan(_token: int) -> None:
        if use_mock_toggle.value:
            _source = MockImageSource(
                image_shape=(256, 256),
                sleep_s=mock_wait_slider.value,
            )
        else:
            _source = real_source()
            if _source is None:
                set_status("Initializing virtual accelerator model…")
                await asyncio.sleep(0)
                from virtual_accelerator.models.staged_model import (
                    get_cu_hxr_staged_model,
                )

                os.environ["LCLS_LATTICE"] = LCLS_LATTICE_PATH
                _model = get_cu_hxr_staged_model()
                _source = ModelImageSource(
                    model=_model,
                    image_pv=IMAGE_PV,
                    xrms_pv=XRMS_PV,
                    yrms_pv=YRMS_PV,
                    reset_values={"track_type": 1},
                )
                set_real_source(_source)
            else:
                set_status("Reusing initialized virtual accelerator model…")
                await asyncio.sleep(0)

            if hasattr(_source, "reset"):
                _source.reset()

        _quad_values = np.linspace(
            scan_min_slider.value,
            scan_max_slider.value,
            int(scan_steps_slider.value),
        )
        _n_steps = len(_quad_values)

        for _i, _qval in enumerate(_quad_values):
            if run_token() != _token or not scan_running():
                return

            set_status(f"Step {_i + 1}/{_n_steps} — **{SCAN_PV}** = `{_qval:.2f}` kG …")

            if getattr(_source, "thread_safe", False):
                _frame = await asyncio.to_thread(_source.get_frame, SCAN_PV, _qval, _i)
            else:
                # Tao/Bmad-backed sources are not thread-safe; keep them on the
                # main thread to avoid pytao parse errors / segmentation faults.
                await asyncio.sleep(0)
                _frame = _source.get_frame(SCAN_PV, _qval, _i)

            if run_token() != _token or not scan_running():
                return

            update_dashboard(_frame)
            await asyncio.sleep(0.05)

        if run_token() == _token:
            set_status(f"Scan complete — {_n_steps} steps finished.")
            set_scan_running(False)

    asyncio.create_task(_run_scan(_token))


# ── Cell 6: stop handler ─────────────────────────────────────────────────────
@app.cell
def stop_scan(stop_button, scan_running, set_scan_running, set_status, mo):
    mo.stop(not stop_button.value or not scan_running())
    set_status("Scan stopped by user.")
    set_scan_running(False)


# ── Cell 7: status callout ───────────────────────────────────────────────────
@app.cell
def status_bar(status_text, mo):
    status_callout = mo.callout(mo.md(status_text()), kind="info")
    status_callout
    return (status_callout,)
