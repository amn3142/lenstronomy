# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`lenstronomy` is a scientific Python package for modeling strong gravitational lenses
(time-delay cosmography, dark matter substructure, galaxy morphology, quasar-host
decomposition, etc.). It is an astropy-affiliated package distributed via PyPI and
conda-forge. Code lives under `lenstronomy/`, tests mirror it 1:1 under `test/`
(e.g. `lenstronomy/LensModel/` ↔ `test/test_LensModel/`).

## Commands

Install in editable/development mode:

```
pip install -r requirements.txt
pip install -r test_requirements.txt
python setup.py develop --user
```

Run the full test suite (pytest picks up `test/` per `setup.cfg`):

```
pytest
```

Run a single test file / test / class:

```
pytest test/test_LensModel/test_lens_model.py
pytest test/test_LensModel/test_lens_model.py::TestLensModel::test_init
pytest test/test_LensModel/test_lens_model.py -k test_init
```

Run with coverage (as CI does):

```
pytest --cov=./ --cov-report=xml
```

Formatting/linting is enforced via pre-commit (black, black-jupyter, docformatter):

```
pre-commit run --all-files
```

Line length follows `black` style (88 chars), not PEP8's 79 — this is an explicit
exception called out in `CONTRIBUTING.rst`.

Build docs (Sphinx, from `docs/`):

```
cd docs && sphinx-build -b html ./ _build/
```

CI (`.github/workflows/ci_test.yml`) runs on Python 3.11–3.13 via plain `pytest`.

## Architecture

lenstronomy composes lens modeling out of independent, string-keyed "profile" plugins,
then assembles them through a small number of layered APIs. Understanding the flow
between these layers matters more than any single file:

1. **Model layer** — `LensModel/`, `LightModel/`, `PointSource/` each define a
   container class (`LensModel`, `LightModel`, `PointSource`) that takes a list of
   model-name strings (e.g. `["EPL", "SHEAR"]`) plus per-model kwargs lists. Each
   container resolves names to profile classes through a base class
   (`ProfileListBase` for lenses, analogous bases for light/point-source) that holds
   a `_SUPPORTED_MODELS` allowlist and a big `if/elif` dispatcher (e.g.
   `lens_class()` in `LensModel/profile_list_base.py`) mapping the string to a
   profile class in `LensModel/Profiles/`, `LightModel/Profiles/`, etc. **Adding a
   new profile means: create the profile class (with the standard profile-class
   interface, e.g. `function`, `derivatives`, `hessian` for lens profiles), then
   register its name in both `_SUPPORTED_MODELS` and the dispatcher.**
   - `LensModel` supports single-plane and multi-plane (`MultiPlane/`) ray tracing,
     line-of-sight corrections (`LineOfSight/`), and a decoupled multi-plane mode for
     speed (`MultiPlane/decoupled_multi_plane.py`).
   - `LightModel` (via `linear_basis.py`) treats surface-brightness amplitudes as
     linear parameters that can be solved for analytically rather than sampled.

2. **Data/imaging layer** — `Data/` (`ImageData`/`PixelGrid`, `PSF`, coordinate
   transforms) describes the observed pixel grid and instrument. `ImSim/image_model.py`
   (`ImageModel`) combines a `Data` instance, a `PSF` instance, and the model-layer
   classes above (lens, source light, lens light, point source, extinction) to render
   a simulated/lensed image, using `ImSim/Numerics/` for PSF convolution and
   sub-pixel/sub-grid ray-shooting strategies. `ImSim/MultiBand/` extends this to
   jointly model multiple bands/images (e.g. `SingleBandMultiModel`).

3. **Simulation API** — `SimulationAPI/` (`sim_api.py`, `model_api.py`,
   `observation_api.py`) is the convenience layer for *generating* mock data: it
   converts astronomical magnitudes/zero-points into the internal surface-brightness
   amplitude convention used by `LightModel`, given telescope/observation configs
   under `SimulationAPI/ObservationConfig/`.

4. **Inference layer** — `Sampling/likelihood.py` (`Likelihood`) builds a posterior
   from one or more `ImageModel`/data sets plus priors; `Sampling/parameters.py`
   (`Param`) handles the mapping between free/fixed/linear parameters and sampler
   vectors. `Sampling/Samplers/` wraps external samplers (emcee, dynesty, MultiNest,
   PolyChord, Nautilus, Cobaya, zeus) behind a common interface driven by
   `Sampling/sampler.py`.

5. **Workflow layer** — `Workflow/fitting_sequence.py` (`FittingSequence`) is the
   top-level orchestrator most users/scripts interact with: given
   `kwargs_data_joint`, `kwargs_model`, `kwargs_constraints`, `kwargs_likelihood`,
   `kwargs_params`, it runs a sequence of fitting steps (PSO, MCMC/nested sampling,
   PSF iteration via `psf_fitting.py`, alignment via `alignment_matching.py`, flux
   calibration via `flux_calibration.py`), updating model kwargs between steps via
   `multi_band_manager.py`/`update_manager.py`. `Util/class_creator.py` is the shared
   factory that turns `kwargs_model` into the actual `LensModel`/`LightModel`/
   `PointSource`/`ImageModel` instances used throughout this stack — most modules
   that need to build "the model classes from kwargs" go through it rather than
   constructing classes directly.

6. **Analysis/post-processing** — `Analysis/` (e.g. `td_cosmography.py` for
   time-delay cosmography, `kinematics_api.py`, `lens_profile.py`) and `GalKin/` /
   `JAMPy/` (kinematics modeling, JAM integration) operate on already-fit model
   results rather than performing the fit itself.

Cross-cutting: `Util/` holds stateless numerical helpers shared across all layers
(`param_util.py` for coordinate/ellipticity conventions, `util.py` grid utilities,
`constants.py`, `numba_util.py` for JIT decorators used by performance-critical
profiles). `Cosmo/lens_cosmo.py` centralizes cosmology-dependent distance/mass
conversions used by both the lens model (multi-plane distances) and analysis modules.

### Conventions worth knowing

- Model components are configured as **parallel lists**: a `_model_list` of name
  strings plus a `kwargs_*` list of dicts in the same order (used identically across
  lens, light, and point-source models, and in `FittingSequence`'s
  `kwargs_params`/`kwargs_model` structures).
- Optional per-profile init kwargs are passed via a `profile_kwargs_list` parallel to
  the model list, not as extra positional args to the container.
- Numba-accelerated profiles (e.g. `epl_numba.py`) coexist with pure-Python versions
  of the same profile; check `Util/numba_util.py` for the JIT wrapper convention
  before assuming numba is mandatory (it degrades gracefully if unavailable).
- lenstronomy core only requires NumPy/SciPy/Astropy at import time; heavier optional
  dependencies (samplers, `slitronomy`, `coolest`, `starred-astro`, `colossus`, JAX
  via `use_jax=True` in `LensModel`) are imported lazily inside the specific modules
  that need them, so don't assume they're globally available.
