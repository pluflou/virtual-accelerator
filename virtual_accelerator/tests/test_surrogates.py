import math

import pytest

from virtual_accelerator.surrogates.injector_surrogate import InjectorSurrogate


def test_injector_surrogate():
    # test to make sure that the surrogate can be
    # initialized and returns an output beam distribution
    surrogate = InjectorSurrogate(n_particles=1000)
    output = surrogate.get(["output_beam"])
    assert "output_beam" in output
    beam = output["output_beam"]
    assert beam.x.shape[0] == 1000

    # check to make sure that changing a control variable changes
    # the output beam distribution
    initial_beam = surrogate.get(["output_beam"])["output_beam"]
    surrogate.set({"QUAD:IN20:525:BCTRL": -5.0})
    updated_beam = surrogate.get(["output_beam"])["output_beam"]
    assert not (initial_beam.x == updated_beam.x).all()
    assert surrogate.get(["QUAD:IN20:525:BCTRL"])["QUAD:IN20:525:BCTRL"] == -5.0


def test_injector_surrogate_outputs_are_physical():
    "Avoids bugs due to YAML/loading errors that can be silent"
    surrogate = InjectorSurrogate(n_particles=1000)

    outputs = surrogate.get(
        [
            "OTRS:IN20:571:XRMS",
            "OTRS:IN20:571:YRMS",
            "sigma_z",
            "norm_emit_x",
            "norm_emit_y",
        ]
    )

    xrms = outputs["OTRS:IN20:571:XRMS"]
    yrms = outputs["OTRS:IN20:571:YRMS"]
    sigma_z = outputs["sigma_z"]
    norm_emit_x = outputs["norm_emit_x"]
    norm_emit_y = outputs["norm_emit_y"]

    assert 0.0 < xrms < 1.0e4
    assert 0.0 < yrms < 1.0e4
    assert 0.0 < sigma_z < 1.0e2
    assert 0.0 < norm_emit_x < 1.0e-3
    assert 0.0 < norm_emit_y < 1.0e-3


def test_injector_surrogate_supports_camr_rms_inputs():
    surrogate = InjectorSurrogate(n_particles=1000)

    supported_variables = surrogate.supported_variables

    assert "CAMR:IN20:186:R_DIST" in supported_variables
    assert "CAMR:IN20:186:XRMS" in supported_variables
    assert "CAMR:IN20:186:YRMS" in supported_variables


def test_injector_surrogate_camr_rms_initial_state_is_valid():
    surrogate = InjectorSurrogate(n_particles=1000)

    xrms = surrogate.get(["CAMR:IN20:186:XRMS"])["CAMR:IN20:186:XRMS"]
    yrms = surrogate.get(["CAMR:IN20:186:YRMS"])["CAMR:IN20:186:YRMS"]

    assert math.isnan(xrms)
    assert math.isnan(yrms)


def test_injector_surrogate_derives_r_dist_from_camr_rms(monkeypatch):
    surrogate = InjectorSurrogate(n_particles=1000)
    captured = {}
    original_set = surrogate.model.set

    def recording_set(values):
        captured.clear()
        captured.update(values)
        return original_set(values)

    monkeypatch.setattr(surrogate.model, "set", recording_set)

    surrogate.set(
        {
            "CAMR:IN20:186:XRMS": 300.0,
            "CAMR:IN20:186:YRMS": 400.0,
        }
    )

    assert "CAMR:IN20:186:XRMS" not in captured
    assert "CAMR:IN20:186:YRMS" not in captured
    assert captured["CAMR:IN20:186:R_DIST"] == pytest.approx(500.0)
    assert surrogate.get(["CAMR:IN20:186:XRMS"])["CAMR:IN20:186:XRMS"] == 300.0
    assert surrogate.get(["CAMR:IN20:186:YRMS"])["CAMR:IN20:186:YRMS"] == 400.0


def test_injector_surrogate_partial_camr_rms_uses_default_r_dist(monkeypatch):
    surrogate = InjectorSurrogate(n_particles=1000)
    default_r_dist = surrogate.get(["CAMR:IN20:186:R_DIST"])["CAMR:IN20:186:R_DIST"]
    surrogate.set({"CAMR:IN20:186:R_DIST": default_r_dist + 10.0})

    captured = {}
    original_set = surrogate.model.set

    def recording_set(values):
        captured.clear()
        captured.update(values)
        return original_set(values)

    monkeypatch.setattr(surrogate.model, "set", recording_set)

    surrogate.set({"CAMR:IN20:186:XRMS": 300.0})

    assert captured["CAMR:IN20:186:R_DIST"] == pytest.approx(default_r_dist)
    assert surrogate.get(["CAMR:IN20:186:R_DIST"])[
        "CAMR:IN20:186:R_DIST"
    ] == pytest.approx(default_r_dist)
    assert math.isnan(surrogate.get(["CAMR:IN20:186:XRMS"])["CAMR:IN20:186:XRMS"])
    assert math.isnan(surrogate.get(["CAMR:IN20:186:YRMS"])["CAMR:IN20:186:YRMS"])
