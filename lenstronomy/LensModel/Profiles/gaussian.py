__author__ = "sibirrer"
# this file contains a class to make a gaussian

import numpy as np
import scipy.special
import scipy.integrate as integrate
from lenstronomy.LensModel.Profiles.gaussian_potential import GaussianPotential
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["Gaussian"]


class Gaussian(LensProfileBase):
    """This class contains functions to evaluate a Gaussian convergence and calculates
    its derivative and hessian matrix."""

    param_names = ["amp", "sigma", "center_x", "center_y"]
    lower_limit_default = {"amp": 0, "sigma": 0, "center_x": -100, "center_y": -100}
    upper_limit_default = {"amp": 100, "sigma": 100, "center_x": 100, "center_y": 100}

    def __init__(self):
        self.gaussian = GaussianPotential()
        self.ds = 0.00001
        super(LensProfileBase, self).__init__()

    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        """Returns potential for a Gaussian convergence.

        :param x: x position
        :param y: y position
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        sigma_x, sigma_y = sigma, sigma
        c = 1.0 / (2 * sigma_x * sigma_y)
        if isinstance(x_, int) or isinstance(x_, float):
            num_int = self._num_integral(r, c)
        else:
            num_int = []
            for i in range(len(x_)):
                num_int.append(self._num_integral(r[i], c))
            num_int = np.array(num_int)
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        amp2d = amp_density / (np.sqrt(np.pi) * np.sqrt(sigma_x * sigma_y * 2))
        amp2d *= 2 * 1.0 / (2 * c)
        return num_int * amp2d

    @staticmethod
    def _num_integral(r, c):
        """Numerical integral (1-e^{-c*x^2})/x dx [0..r]

        :param r: radius
        :param c: 1/2sigma^2
        :return:
        """
        if r == 0:
            return 0
        out = integrate.quad(lambda x: (1 - np.exp(-c * x**2)) / x, 0, r)
        return out[0]

    def derivatives(self, x, y, amp, sigma, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function.

        :param x: x position
        :param y: y position
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        sigma_x, sigma_y = sigma, sigma
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, self.ds)
        else:
            R[R <= self.ds] = self.ds
        alpha = self.alpha_abs(R, amp, sigma)
        return alpha / R * x_, alpha / R * y_

    def hessian(self, x, y, amp, sigma, center_x=0, center_y=0):
        """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2.

        :param x: x position
        :param y: y position
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        sigma_x, sigma_y = sigma, sigma
        if isinstance(r, int) or isinstance(r, float):
            r = max(r, self.ds)
        else:
            r[r <= self.ds] = self.ds
        d_alpha_dr = -self.d_alpha_dr(r, amp, sigma_x, sigma_y)
        alpha = self.alpha_abs(r, amp, sigma)

        f_xx = -(d_alpha_dr / r + alpha / r**2) * x_**2 / r + alpha / r
        f_yy = -(d_alpha_dr / r + alpha / r**2) * y_**2 / r + alpha / r
        f_xy = -(d_alpha_dr / r + alpha / r**2) * x_ * y_ / r
        return f_xx, f_xy, f_xy, f_yy

    def density(self, r, amp, sigma):
        """3d mass density as a function of radius r.

        :param r: radius
        :param amp: 3d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        return self.gaussian.function(r, 0, amp, sigma_x, sigma_y)

    def density_lens(self, r, amp, sigma):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity. (optional definition)

        .. math::
            \\kappa(x, y) = \\int_{-\\infty}^{\\infty} \\rho(x, y, z) dz

        :param r: radial distance from the center (in 3D)
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        :return: density
        """
        amp_density = self._amp2d_to_3d(amp, sigma, sigma)
        return self.density(r, amp_density, sigma)

    def density_2d(self, x, y, amp, sigma, center_x=0, center_y=0):
        """Projected 2d density at position (x,y)

        :param x: x position
        :param y: y position
        :param amp: 3d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        sigma_x, sigma_y = sigma, sigma
        amp2d = self._amp3d_to_2d(amp, sigma_x, sigma_y)
        return self.gaussian.function(x, y, amp2d, sigma_x, sigma_y, center_x, center_y)

    def mass_2d(self, R, amp, sigma):
        """Mass enclosed in a circle of radius R when projected into 2d.

        :param R: projected radius
        :param amp: 3d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        amp2d = amp / (np.sqrt(np.pi) * np.sqrt(sigma_x * sigma_y * 2))
        c = 1.0 / (2 * sigma_x * sigma_y)
        return amp2d * 2 * np.pi * 1.0 / (2 * c) * (1.0 - np.exp(-c * R**2))

    def mass_2d_lens(self, R, amp, sigma):
        """Mass enclosed in a circle of radius R when projected into 2d.

        :param R: projected radius
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        return self.mass_2d(R, amp_density, sigma)

    def alpha_abs(self, R, amp, sigma):
        """Absolute value of the deflection.

        :param R: radius projected into 2d
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        :return:
        """
        sigma_x, sigma_y = sigma, sigma
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        alpha = self.mass_2d(R, amp_density, sigma) / np.pi / R
        return alpha

    def d_alpha_dr(self, R, amp, sigma_x, sigma_y):
        """Derivative of deflection angle w.r.t r.

        :param R: radius projected into 2d
        :param amp: 2d amplitude of Gaussian
        :param sigma_x: standard deviation of Gaussian in x direction
        :param sigma_y: standard deviation of Gaussian in y direction
        """
        c = 1.0 / (2 * sigma_x * sigma_y)
        A = self._amp2d_to_3d(amp, sigma_x, sigma_y) * np.sqrt(
            2 / np.pi * sigma_x * sigma_y
        )
        return 1.0 / R**2 * (-1 + (1 + 2 * c * R**2) * np.exp(-c * R**2)) * A

    def mass_3d(self, R, amp, sigma):
        """Mass enclosed within a 3D sphere of projected radius R given a lens
        parameterization with angular units. The input parameter amp is the 3d
        amplitude.

        :param R: radius projected into 2d
        :param amp: 3d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        A = amp / (2 * np.pi * sigma_x * sigma_y)
        c = 1.0 / (2 * sigma_x * sigma_y)
        result = (
            1.0
            / (2 * c)
            * (
                -R * np.exp(-c * R**2)
                + scipy.special.erf(np.sqrt(c) * R) * np.sqrt(np.pi / (4 * c))
            )
        )
        return result * A * 4 * np.pi

    def mass_3d_lens(self, R, amp, sigma):
        """Mass enclosed within a 3D sphere of projected radius R given a lens
        parameterization with angular units. The input parameters are identical as for
        the derivatives definition. (optional definition)

        :param R: radius projected into 2d
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        return self.mass_3d(R, amp_density, sigma)

    @staticmethod
    def _amp3d_to_2d(amp, sigma_x, sigma_y):
        """Converts 3d density into 2d density parameter.

        :param amp: 3d amplitude of Gaussian
        :param sigma_x: standard deviation of Gaussian in x direction
        :param sigma_y: standard deviation of Gaussian in y direction
        """
        return amp * np.sqrt(np.pi) * np.sqrt(sigma_x * sigma_y * 2)

    @staticmethod
    def _amp2d_to_3d(amp, sigma_x, sigma_y):
        """Converts 2d density into 3d density parameter.

        :param amp: 2d amplitude of Gaussian
        :param sigma_x: standard deviation of Gaussian in x direction
        :param sigma_y: standard deviation of Gaussian in y direction
        """
        return amp / (np.sqrt(np.pi) * np.sqrt(sigma_x * sigma_y * 2))
