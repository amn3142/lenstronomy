import lenstronomy.Util.kernel_util as kernel_util

__all__ = ["AnalyticPSFModel", "AO_PSF_MODEL"]


class AnalyticPSFModel(object):
    """Bundles a PSF-kernel-generating function together with the names and default
    fit bounds of its shape parameters, as one object.

    This is the unit that gets swapped to change the analytic PSF functional form: the
    same instance is passed as ``kwargs_psf['psf_model']`` when constructing a band's
    PSF() instance, and its ``param_names``/``lower_limit_default``/
    ``upper_limit_default`` are read generically (never hardcoded) wherever the
    analytic PSF is fit. No other lenstronomy code needs to change to switch to a
    different functional form -- only which ``AnalyticPSFModel`` instance a script
    points at.
    """

    def __init__(self, function, param_names, lower_limit_default, upper_limit_default):
        """

        :param function: callable(num_pix, delta_pix, **kwargs) -> 2d numpy array,
            matching the signature convention of
            lenstronomy.Util.kernel_util.kernel_gaussian. Must be a module-level
            (importable) callable -- not a lambda or closure -- to remain picklable
            for disk-based multi_band_list workflows.
        :param param_names: list of str, names of the free shape parameters (excludes
            num_pix/delta_pix; kernels are always centered and unit-flux-normalized)
        :param lower_limit_default: dict, name -> default lower bound
        :param upper_limit_default: dict, name -> default upper bound
        """
        self.param_names = list(param_names)
        self.lower_limit_default = dict(lower_limit_default)
        self.upper_limit_default = dict(upper_limit_default)
        self._function = function

    def function(self, num_pix, delta_pix, **kwargs):
        """

        :param num_pix: number of pixels (odd)
        :param delta_pix: pixel scale
        :param kwargs: values for each name in self.param_names
        :return: 2d numpy array kernel
        """
        return self._function(num_pix, delta_pix, **kwargs)


AO_PSF_MODEL = AnalyticPSFModel(
    function=kernel_util.ao_psf_kernel,
    param_names=["fwhm_core", "fwhm_halo", "strehl", "e1_halo", "e2_halo"],
    lower_limit_default={
        "fwhm_core": 0.01,
        "fwhm_halo": 0.1,
        "strehl": 0.01,
        "e1_halo": -0.6,
        "e2_halo": -0.6,
    },
    upper_limit_default={
        "fwhm_core": 0.3,
        "fwhm_halo": 3.0,
        "strehl": 1.0,
        "e1_halo": 0.6,
        "e2_halo": 0.6,
    },
)
