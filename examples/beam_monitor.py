"""
beam_monitor.py
---------------
Swappable ImageSource abstraction for the live beam monitor marimo notebook.

Define a new class that implements the `ImageSource` protocol and pass it to the
marimo notebook to swap out the data source (e.g. real EPICS, file replay, etc.)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data container returned by every source on each "shot"
# ---------------------------------------------------------------------------


@dataclass
class BeamFrame:
    """One snapshot of beam data."""

    # Scalar diagnostics
    xrms: float = 0.0  # µm
    yrms: float = 0.0  # µm
    sigma_z: float = 0.0  # m
    norm_emit_x: float = 0.0  # m·rad
    norm_emit_y: float = 0.0  # m·rad

    # 2-D OTR image (nRow × nCol), float64
    image: Optional[np.ndarray] = None

    # Phase-space scatter arrays (m)
    beam_x: Optional[np.ndarray] = None
    beam_px: Optional[np.ndarray] = None

    # Twiss arrays along the lattice
    twiss_s: Optional[np.ndarray] = None
    twiss_a_beta: Optional[np.ndarray] = None
    twiss_b_beta: Optional[np.ndarray] = None

    # Metadata
    scan_pv: str = ""
    scan_value: float = float("nan")
    step_index: int = 0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Abstract base class – implement this to swap sources
# ---------------------------------------------------------------------------


class ImageSource(ABC):
    """
    Protocol for a live beam data source.

    Implementations must be safe to call from an asyncio task. Some sources
    can be offloaded to a worker thread, while others (notably Tao/Bmad-backed
    model sources) must stay on the main thread.
    """

    thread_safe: bool = False

    @abstractmethod
    def get_frame(self, scan_pv: str, scan_value: float, step_index: int) -> BeamFrame:
        """
        Set *scan_pv* to *scan_value*, run the model / read hardware,
        and return a populated :class:`BeamFrame`.

        This method **may block** – the marimo notebook wraps it in
        ``asyncio.to_thread`` so the UI stays responsive.
        """
        ...

    def reset(self) -> None:
        """Optional: reset any internal state between scans."""


# ---------------------------------------------------------------------------
# Concrete implementation: virtual-accelerator StagedModel
# ---------------------------------------------------------------------------


class ModelImageSource(ImageSource):
    """
    Use the SLAC virtual-accelerator :class:`StagedModel` as the data source.

    Parameters
    ----------
    model:
        A ready-to-use ``StagedModel`` (or any ``LUMEModel``) instance.
    image_pv:
        PV name for the 2-D OTR image, e.g. ``"OTRS:IN20:711:Image:ArrayData"``.
    xrms_pv:
        PV name for horizontal RMS beam size (µm).
    yrms_pv:
        PV name for vertical RMS beam size (µm).
    particle_source:
        PV name of the ``ParticleGroup`` variable used for the phase-space scatter.
        Defaults to ``"output_beam"``.
    max_scatter_points:
        Downsample the scatter plot to at most this many points for speed.
    """

    thread_safe = False

    def __init__(
        self,
        model,
        image_pv: str = "OTRS:IN20:711:Image:ArrayData",
        xrms_pv: str = "OTRS:IN20:571:XRMS",
        yrms_pv: str = "OTRS:IN20:571:YRMS",
        # sigma_z_pv: str = "sigma_z",
        # norm_emit_x_pv: str = "norm_emit_x",
        # norm_emit_y_pv: str = "norm_emit_y",
        particle_source: str = "OTR4_beam",
        max_scatter_points: int = 3000,
        reset_values: Optional[dict[str, object]] = None,
        twiss_s_pv: str = "s",
        twiss_a_beta_pv: str = "a.beta",
        twiss_b_beta_pv: str = "b.beta",
    ):
        self.model = model
        self.image_pv = image_pv
        self.xrms_pv = xrms_pv
        self.yrms_pv = yrms_pv
        self.particle_source = particle_source
        self.max_scatter_points = max_scatter_points
        self.reset_values = reset_values or {}
        self.twiss_s_pv = twiss_s_pv
        self.twiss_a_beta_pv = twiss_a_beta_pv
        self.twiss_b_beta_pv = twiss_b_beta_pv

    def get_frame(self, scan_pv: str, scan_value: float, step_index: int) -> BeamFrame:
        self.model.set({scan_pv: scan_value})

        pvs = [
            self.image_pv,
            self.xrms_pv,
            self.yrms_pv,
            # self.sigma_z_pv,
            # self.norm_emit_x_pv,
            # self.norm_emit_y_pv,
            self.particle_source,
            self.twiss_s_pv,
            self.twiss_a_beta_pv,
            self.twiss_b_beta_pv,
        ]
        result = self.model.get(pvs)

        image = result.get(self.image_pv)
        beam = result.get(self.particle_source)
        xrms = beam["sigma_x"] * 1e6  # m -> µm
        yrms = beam["sigma_y"] * 1e6  # m -> µm
        sigma_z = beam["sigma_z"]  # already in m
        norm_emit_x = beam["norm_emit_x"]
        norm_emit_y = beam["norm_emit_y"]

        twiss_s = result.get(self.twiss_s_pv)
        twiss_a_beta = result.get(self.twiss_a_beta_pv)
        twiss_b_beta = result.get(self.twiss_b_beta_pv)

        # Downsample scatter for rendering performance
        bx = bpx = None
        if beam is not None:
            x = np.asarray(beam["x"])
            px = np.asarray(beam["px"])
            n = len(x)
            if n > self.max_scatter_points:
                idx = np.random.choice(n, self.max_scatter_points, replace=False)
                x, px = x[idx], px[idx]
            bx, bpx = x * 1e6, px  # x: m -> um, px: eV/c

        return BeamFrame(
            xrms=float(xrms),
            yrms=float(yrms),
            sigma_z=float(sigma_z),
            norm_emit_x=float(norm_emit_x),
            norm_emit_y=float(norm_emit_y),
            image=image,
            beam_x=bx,
            beam_px=bpx,
            twiss_s=None if twiss_s is None else np.asarray(twiss_s, dtype=float),
            twiss_a_beta=None
            if twiss_a_beta is None
            else np.asarray(twiss_a_beta, dtype=float),
            twiss_b_beta=None
            if twiss_b_beta is None
            else np.asarray(twiss_b_beta, dtype=float),
            scan_pv=scan_pv,
            scan_value=scan_value,
            step_index=step_index,
            timestamp=time.time(),
        )

    def reset(self) -> None:
        """
        Soft-reset the wrapped model for repeated scans.

        We intentionally avoid calling ``self.model.reset()`` here because the
        cached Tao/Bmad model may carry read-only screen/image PVs in its
        internal initial state, and replaying that state raises on reset.
        For the monitor app we only need to restore a small set of writable
        controls (for example ``track_type``), which are supplied via
        ``reset_values``.
        """
        if self.reset_values:
            self.model.set(self.reset_values)


# ---------------------------------------------------------------------------
# Concrete implementation: synthetic / mock source (no hardware required)
# ---------------------------------------------------------------------------


class MockImageSource(ImageSource):
    """
    Generates synthetic beam images and scalars for development / CI.

    The beam spot is a 2-D Gaussian whose sigma_x grows with |scan_value|,
    mimicking a real quadrupole scan.

    Parameters
    ----------
    image_shape:
        (nRow, nCol) of the fake OTR image.
    n_particles:
        Number of particles in the phase-space scatter.
    noise_level:
        Gaussian noise amplitude added to the image.
    sleep_s:
        Artificial latency per step to simulate computation time.
    """

    thread_safe = True

    def __init__(
        self,
        image_shape: tuple[int, int] = (256, 256),
        n_particles: int = 3000,
        noise_level: float = 0.02,
        sleep_s: float = 0.2,
    ):
        self.image_shape = image_shape
        self.n_particles = n_particles
        self.noise_level = noise_level
        self.sleep_s = sleep_s

    def get_frame(self, scan_pv: str, scan_value: float, step_index: int) -> BeamFrame:
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)

        nrow, ncol = self.image_shape
        cy, cx = nrow / 2, ncol / 2

        # Spot size varies with quad strength
        sigma_x_px = max(4.0, 20.0 + 2.5 * scan_value)
        sigma_y_px = max(4.0, 20.0 - 1.5 * scan_value)

        y_idx, x_idx = np.mgrid[0:nrow, 0:ncol]
        image = np.exp(
            -0.5 * ((x_idx - cx) / sigma_x_px) ** 2
            - 0.5 * ((y_idx - cy) / sigma_y_px) ** 2
        )
        image += np.random.normal(0, self.noise_level, image.shape)
        image = np.clip(image, 0, None)

        # Scalar diagnostics
        res_um = 12.0  # µm/px (approximate)
        xrms = sigma_x_px * res_um
        yrms = sigma_y_px * res_um
        norm_emit_x = max(1e-7, 5e-7 + scan_value * 2e-8)
        norm_emit_y = max(1e-7, 5e-7 - scan_value * 1.5e-8)
        sigma_z = 4.6e-4

        # Phase-space scatter
        bx = np.random.normal(0, xrms, self.n_particles)
        bpx = np.random.normal(0, yrms, self.n_particles)

        # Simple synthetic Twiss functions along s
        twiss_s = np.linspace(0.0, 40.0, 200)
        twiss_a_beta = 6.0 + 1.5 * np.sin(twiss_s / 6.0 + 0.15 * scan_value)
        twiss_b_beta = 8.0 + 2.0 * np.cos(twiss_s / 7.5 - 0.12 * scan_value)

        return BeamFrame(
            xrms=xrms,
            yrms=yrms,
            sigma_z=sigma_z,
            norm_emit_x=norm_emit_x,
            norm_emit_y=norm_emit_y,
            image=image,
            beam_x=bx,
            beam_px=bpx,
            twiss_s=twiss_s,
            twiss_a_beta=twiss_a_beta,
            twiss_b_beta=twiss_b_beta,
            scan_pv=scan_pv,
            scan_value=scan_value,
            step_index=step_index,
            timestamp=time.time(),
        )
