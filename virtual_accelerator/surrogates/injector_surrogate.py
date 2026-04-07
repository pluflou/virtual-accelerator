import copy
from pathlib import Path
import os
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from cheetah.particles import ParticleBeam
from lume.model import LUMEModel
from lume.variables import ParticleGroupVariable
from lume_torch.base import LUMETorchModel
from lume_torch.models.torch_model import TorchModel
from scipy import constants

OTR2_BEAM_ENERGY = 135.0e6  # eV
CAMR_R_DIST_VARIABLE = "CAMR:IN20:186:R_DIST"
CAMR_XRMS_VARIABLE = "CAMR:IN20:186:XRMS"
CAMR_YRMS_VARIABLE = "CAMR:IN20:186:YRMS"
UNSET_CAMR_RMS_VALUE = float("nan")


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Return a NumPy view/copy from tensor-like input on CPU without gradients."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_python_scalar(value: Any, key: str) -> Any:
    """Convert one-element tensors to scalars while preserving non-tensors."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.numel() != 1:
        raise ValueError(
            f"Expected scalar tensor for cache key '{key}', got shape {tuple(value.shape)}"
        )
    return value.item()


def _copy_variable_with_name(variable: Any, name: str) -> Any:
    """Return a shallow copy of a variable-like object with an updated name."""
    cloned = copy.copy(variable)
    if hasattr(cloned, "name"):
        cloned.name = name
    return cloned


def _compute_r_dist(x_rms: Any, y_rms: Any) -> Any:
    """Compute the compound CAMR radial RMS using NumPy or torch semantics."""
    if isinstance(x_rms, torch.Tensor) or isinstance(y_rms, torch.Tensor):
        x_tensor = x_rms if isinstance(x_rms, torch.Tensor) else torch.as_tensor(x_rms)
        y_tensor = (
            y_rms.to(dtype=x_tensor.dtype, device=x_tensor.device)
            if isinstance(y_rms, torch.Tensor)
            else torch.as_tensor(y_rms, dtype=x_tensor.dtype, device=x_tensor.device)
        )
        return torch.sqrt(x_tensor.square() + y_tensor.square())

    r_dist = np.sqrt(np.asarray(x_rms) ** 2 + np.asarray(y_rms) ** 2)
    return r_dist.item() if np.ndim(r_dist) == 0 else r_dist


def to_openpmd_particlegroup(beam) -> "openpmd.ParticleGroup":  # noqa: F821
    """
    Convert the `ParticleBeam` to an openPMD `ParticleGroup` object.

    NOTE: openPMD uses boolean particle status flags, i.e. alive or dead. Cheetah's
        survival probabilities are converted to status flags by thresholding at 0.5.

    NOTE: At the moment this method only supports non-vectorised particle
        distributions.

    :return: openPMD `ParticleGroup` object with the `ParticleBeam`'s particles.
    """
    try:
        import pmd_beamphysics as openpmd
    except ImportError:
        raise ImportError(
            """To use the openPMD beam export, openPMD-beamphysics must be
            installed."""
        )

    # For now only support non-vectorised particle distributions
    if len(beam.particles.shape) != 2:
        raise ValueError("Only non-vectorised particle distributions are supported.")

    px = beam.px * beam.p0c
    py = beam.py * beam.p0c
    p_total = (beam.energies.square() - beam.species.mass_eV.square()).sqrt()
    pz = (p_total.square() - px.square() - py.square()).sqrt()
    t = beam.tau / constants.speed_of_light
    # TODO: To be discussed
    status = beam.survival_probabilities > 0.5

    data = {
        "x": _tensor_to_numpy(beam.x),
        "y": _tensor_to_numpy(beam.y),
        "z": _tensor_to_numpy(beam.tau),
        "px": _tensor_to_numpy(px),
        "py": _tensor_to_numpy(py),
        "pz": _tensor_to_numpy(pz),
        "t": _tensor_to_numpy(t),
        "weight": _tensor_to_numpy(
            -beam.particle_charges
        ),  # need to make at least 1d and negate
        "status": _tensor_to_numpy(status.int()),  # need int
        "species": beam.species.name,
    }
    particle_group = openpmd.ParticleGroup(data=data)

    return particle_group


def create_beam_distribution_from_state(
    state: Mapping[str, Any], n_particles: int
) -> ParticleBeam:
    sigma_x = torch.tensor(state["OTRS:IN20:571:XRMS"] * 1e-6)
    sigma_y = torch.tensor(state["OTRS:IN20:571:YRMS"] * 1e-6)
    sigma_z = torch.tensor(state["sigma_z"] * 1e-6)
    normalized_emittance_x = torch.tensor(state["norm_emit_x"])
    normalized_emittance_y = torch.tensor(state["norm_emit_y"])
    energy = OTR2_BEAM_ENERGY
    relativistic_gamma = energy / (
        constants.value("electron mass energy equivalent in MeV") * 1e6
    )
    beam = ParticleBeam.from_twiss(
        num_particles=n_particles,
        beta_x=sigma_x**2 / (normalized_emittance_x / relativistic_gamma),
        beta_y=sigma_y**2 / (normalized_emittance_y / relativistic_gamma),
        alpha_x=torch.tensor(0.1333896),
        alpha_y=torch.tensor(0.1333896),
        emittance_x=normalized_emittance_x / relativistic_gamma,
        emittance_y=normalized_emittance_y / relativistic_gamma,
        sigma_tau=sigma_z,
        energy=torch.tensor(energy),
    )
    beam.particles = beam.particles.squeeze()
    return beam


class InjectorSurrogate(LUMEModel):
    """LUME wrapper around injector torch surrogate with openPMD beam output."""

    # Config path relative to the project root (used when running from source)
    _SOURCE_RELATIVE = (
        Path("subtrees") / "lcls_cu_injector_ml_model" / "model_config.yaml"
    )

    # Config keys whose values are resource paths that need resolving
    _RESOURCE_KEYS = ("model", "input_transformers", "output_transformers")
    _ADAPTER_INPUT_VARIABLES = (CAMR_XRMS_VARIABLE, CAMR_YRMS_VARIABLE)

    @classmethod
    def _candidate_config_roots(cls) -> list[Path]:
        """Return candidate root directories used to locate model config."""
        roots: list[Path] = []

        # Module location (source checkout or installed package layout)
        roots.extend(Path(__file__).resolve().parents)

        # GitHub Actions checkout root
        workspace = os.environ.get("GITHUB_WORKSPACE")
        if workspace:
            roots.append(Path(workspace).resolve())

        # Current working directory and its ancestors
        cwd = Path.cwd().resolve()
        roots.append(cwd)
        roots.extend(cwd.parents)

        # Deduplicate while preserving order
        seen: set[Path] = set()
        unique_roots: list[Path] = []
        for root in roots:
            if root not in seen:
                seen.add(root)
                unique_roots.append(root)
        return unique_roots

    @classmethod
    def _find_config(cls) -> Path:
        """Locate ``model_config.yaml`` regardless of install mode."""
        for root in cls._candidate_config_roots():
            candidate = root / cls._SOURCE_RELATIVE
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            "Could not find model_config.yaml. Looked for "
            f"{cls._SOURCE_RELATIVE} from module/cwd/workspace roots. "
            "Ensure the subtree exists in the checkout, e.g. "
            "'git subtree add --prefix subtrees/lcls_cu_injector_ml_model <remote> <ref>'."
        )

    def __init__(self, n_particles: int = 10000) -> None:
        """Initialize surrogate model and internal cache copy.

        Resource paths inside ``model_config.yaml`` are relative to the
        submodule directory.  A temporary config file with those paths
        rewritten to absolute paths is passed to ``TorchModel`` so that
        initialization succeeds regardless of the current working directory.
        """
        super().__init__()
        tm = self._load_torch_model()
        self.model = LUMETorchModel(tm)
        self.n_particles = n_particles
        self._cache: dict[str, Any] = {
            CAMR_XRMS_VARIABLE: UNSET_CAMR_RMS_VALUE,
            CAMR_YRMS_VARIABLE: UNSET_CAMR_RMS_VALUE,
        }
        self.set({})  # Initializing with defaults of NN model
        self._default_model_values = {
            key: self._cache[key] for key in self.model.supported_variables.keys()
        }

    @classmethod
    def _resolve_resource_paths(cls, config: dict, base_dir: Path) -> dict:
        """Return a copy of config with resource paths made absolute."""
        resolved = dict(config)
        for key in cls._RESOURCE_KEYS:
            if key not in resolved:
                continue
            value = resolved[key]
            if isinstance(value, str):
                resolved[key] = str((base_dir / value).resolve())
            elif isinstance(value, list):
                resolved[key] = [str((base_dir / v).resolve()) for v in value]
        return resolved

    @classmethod
    def _load_torch_model(cls) -> TorchModel:
        """Load :class:`TorchModel` with all resource paths resolved.

        Writes a temporary config YAML whose resource paths are absolute so
        that ``TorchModel`` can locate them regardless of the working directory.
        The temporary file is removed after loading.
        """
        config_path = cls._find_config()
        base_dir = config_path.parent

        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        resolved_config = cls._resolve_resource_paths(config, base_dir)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(resolved_config, tmp, sort_keys=False)
            tmp_path = Path(tmp.name)

        try:
            return TorchModel(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    def _get(self, names: Iterable[str]) -> dict[str, Any]:
        return {name: self._cache[name] for name in names}

    def _prepare_model_values(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Translate adapter inputs into TorchModel inputs before dispatch."""
        model_values = {
            key: value
            for key, value in values.items()
            if key not in self._ADAPTER_INPUT_VARIABLES
        }
        has_xrms = CAMR_XRMS_VARIABLE in values
        has_yrms = CAMR_YRMS_VARIABLE in values

        if has_xrms and has_yrms:
            model_values[CAMR_R_DIST_VARIABLE] = _compute_r_dist(
                values[CAMR_XRMS_VARIABLE], values[CAMR_YRMS_VARIABLE]
            )
        elif has_xrms or has_yrms:
            model_values.setdefault(
                CAMR_R_DIST_VARIABLE, self._default_model_values[CAMR_R_DIST_VARIABLE]
            )

        return model_values

    def _set(self, values: Mapping[str, Any]) -> None:
        """Update model state and regenerate exported output beam."""
        has_xrms = CAMR_XRMS_VARIABLE in values
        has_yrms = CAMR_YRMS_VARIABLE in values

        # Write non-adapter inputs to cache
        for name, value in values.items():
            if name not in self._ADAPTER_INPUT_VARIABLES:
                self._cache[name] = value

        # Update adapter cache based on what was provided
        if has_xrms and has_yrms:
            self._cache[CAMR_XRMS_VARIABLE] = values[CAMR_XRMS_VARIABLE]
            self._cache[CAMR_YRMS_VARIABLE] = values[CAMR_YRMS_VARIABLE]
        elif has_xrms or has_yrms or CAMR_R_DIST_VARIABLE in values:
            self._cache[CAMR_XRMS_VARIABLE] = UNSET_CAMR_RMS_VALUE
            self._cache[CAMR_YRMS_VARIABLE] = UNSET_CAMR_RMS_VALUE

        self.model.set(self._prepare_model_values(values))
        self.update_state()

    @property
    def supported_variables(self) -> dict[str, Any]:
        """Return supported variables without mutating wrapped model metadata."""
        variables = dict(self.model.supported_variables)
        r_dist_variable = variables.get(CAMR_R_DIST_VARIABLE)
        if r_dist_variable is not None:
            variables[CAMR_XRMS_VARIABLE] = _copy_variable_with_name(
                r_dist_variable, CAMR_XRMS_VARIABLE
            )
            variables[CAMR_YRMS_VARIABLE] = _copy_variable_with_name(
                r_dist_variable, CAMR_YRMS_VARIABLE
            )
        variables["output_beam"] = ParticleGroupVariable(
            name="output_beam", read_only=True
        )
        return variables

    def reset(self):
        self.model.reset()
        self._cache = {
            CAMR_XRMS_VARIABLE: UNSET_CAMR_RMS_VALUE,
            CAMR_YRMS_VARIABLE: UNSET_CAMR_RMS_VALUE,
        }
        self.update_state()

    def update_state(self):
        adapter_inputs = {
            key: self._cache.get(key, UNSET_CAMR_RMS_VALUE)
            for key in self._ADAPTER_INPUT_VARIABLES
        }
        model_state = self.model.get(list(self.model.supported_variables.keys()))
        self._cache.update({k: _to_python_scalar(v, k) for k, v in model_state.items()})
        self._cache.update(adapter_inputs)
        beam = create_beam_distribution_from_state(self._cache, self.n_particles)
        self._cache["output_beam"] = to_openpmd_particlegroup(beam)
