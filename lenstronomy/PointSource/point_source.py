import numpy as np
import copy
from lenstronomy.PointSource.point_source_cached import PointSourceCached

__all__ = ["PointSource"]

_SUPPORTED_MODELS = ["UNLENSED", "LENSED_POSITION", "SOURCE_POSITION"]


class PointSource(object):
    def __init__(
        self,
        point_source_type_list,
        lens_model=None,
        fixed_magnification_list=None,
        additional_images_list=None,
        flux_from_point_source_list=None,
        magnification_limit=None,
        save_cache=False,
        kwargs_lens_eqn_solver=None,
        index_lens_model_list=None,
        point_source_frame_list=None,
        redshift_list=None,
    ):
        """

        :param point_source_type_list: list of point source types
        :param lens_model: instance of the LensModel() class
        :param fixed_magnification_list: list of booleans (same length as point_source_type_list).
            If True, magnification ratio of point sources is fixed to the one given by the lens model.
            This option then requires to provide a 'source_amp' amplitude of the source brightness instead of
            'point_amp' the list of image brightnesses.
        :param additional_images_list: list of booleans (same length as point_source_type_list). If True, search for
            additional images of the same source is conducted.
        :param flux_from_point_source_list: list of booleans (optional), if set, will only return image positions
            (for imaging modeling) for the subset of the point source lists that =True. This option enables to model
            imaging data with transient point sources, when the point source positions are measured and present at a
            different time than the imaging data, or when the image position is not known (such as for lensed GW)
        :param magnification_limit: float >0 or None, if float is set and additional images are computed, only those
            images will be computed that exceed the lensing magnification (absolute value) limit
        :param save_cache: bool, saves image positions and only if delete_cache is executed, a new solution of the lens
            equation is conducted with the lens model parameters provided. This can increase the speed as multiple times
            the image positions are requested for the same lens model. Attention in usage!
        :param kwargs_lens_eqn_solver: keyword arguments specifying the numerical settings for the lens equation solver
            see LensEquationSolver() class for details, such as:
            min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100
        :param index_lens_model_list: list (length of different patches/bands) of integer lists, evaluating a subset of
            the lens models per individual bands. e.g., [[0], [2, 3], [1]] assigns the 0th lens model to the 0th band,
            the 2nd and 3rd lens models to the 1st band, and the 1st lens model to the 2nd band.
            If this keyword is set, the image positions need to have a specified band/frame assigned to it
        :param point_source_frame_list: list of list of ints, assigns each model in point_source_type_list a frame list.
            Only relevent for LENSED_POSITION. e.g. if point_source_type_list = ["UNLENSED", "LENSED_POSITION", "LENSED_POSITION"]
            with point_source_frame_list = [None, [0, 1, 2], [1, 2, 0, 1]], then the first LENSED_POSITION will have a frame list of
            [0, 1, 2] and the second LENSED_POSITION will have a frame list of [1, 2, 0, 1]. See docstring for point_source_frame_list
            in PSBase for further details.
        :param redshift_list: list of redshifts (only required for multiple source redshifts)
        :type redshift_list: None or list
        """
        if "LENSED_POSITION" in point_source_type_list:
            if index_lens_model_list is not None and point_source_frame_list is None:
                raise ValueError(
                    "with specified index_lens_model_list, a specified point_source_frame_list argument is "
                    "required for LENSED_POSITION"
                )
            if index_lens_model_list is None:
                point_source_frame_list = [None] * len(point_source_type_list)
        self._index_lens_model_list = index_lens_model_list
        self._point_source_frame_list = point_source_frame_list
        self._lens_model = lens_model
        self.point_source_type_list = point_source_type_list
        self._point_source_list = []
        if fixed_magnification_list is None:
            fixed_magnification_list = [False] * len(point_source_type_list)
        self._fixed_magnification_list = fixed_magnification_list
        if additional_images_list is None:
            additional_images_list = [False] * len(point_source_type_list)
        if flux_from_point_source_list is None:
            flux_from_point_source_list = [True] * len(point_source_type_list)
        self._flux_from_point_source_list = flux_from_point_source_list
        if redshift_list is None:
            redshift_list = [None] * len(point_source_type_list)
        self._redshift_list = redshift_list
        for i, model in enumerate(point_source_type_list):
            if model == "UNLENSED":
                from lenstronomy.PointSource.Types.unlensed import Unlensed

                self._point_source_list.append(
                    PointSourceCached(Unlensed(), save_cache=save_cache)
                )
            elif model == "LENSED_POSITION":
                from lenstronomy.PointSource.Types.lensed_position import (
                    LensedPositions,
                )

                self._point_source_list.append(
                    PointSourceCached(
                        LensedPositions(
                            lens_model,
                            fixed_magnification=fixed_magnification_list[i],
                            additional_images=additional_images_list[i],
                            index_lens_model_list=index_lens_model_list,
                            point_source_frame_list=point_source_frame_list[i],
                            redshift=redshift_list[i],
                        ),
                        save_cache=save_cache,
                    )
                )
            elif model == "SOURCE_POSITION":
                from lenstronomy.PointSource.Types.source_position import (
                    SourcePositions,
                )

                self._point_source_list.append(
                    PointSourceCached(
                        SourcePositions(
                            lens_model,
                            fixed_magnification=fixed_magnification_list[i],
                            redshift=redshift_list[i],
                        ),
                        save_cache=save_cache,
                    )
                )
            else:
                raise ValueError(
                    "Point-source model %s not available. Supported models are %s ."
                    % (model, _SUPPORTED_MODELS)
                )
        if kwargs_lens_eqn_solver is None:
            kwargs_lens_eqn_solver = {}
        self._kwargs_lens_eqn_solver = kwargs_lens_eqn_solver
        self._magnification_limit = magnification_limit
        self._save_cache = save_cache

    def update_search_window(
        self,
        search_window,
        x_center,
        y_center,
        min_distance=None,
        only_from_unspecified=False,
    ):
        """Update the search area for the lens equation solver.

        :param search_window: search_window: window size of the image position search
            with the lens equation solver.
        :param x_center: center of search window
        :param y_center: center of search window
        :param min_distance: minimum search distance
        :param only_from_unspecified: bool, if True, only sets keywords that previously
            have not been set
        :return: updated self instances
        """
        if (
            min_distance is not None
            and "min_distance" not in self._kwargs_lens_eqn_solver
            and only_from_unspecified
        ):
            self._kwargs_lens_eqn_solver["min_distance"] = min_distance
        if only_from_unspecified:
            self._kwargs_lens_eqn_solver["search_window"] = (
                self._kwargs_lens_eqn_solver.get("search_window", search_window)
            )
            self._kwargs_lens_eqn_solver["x_center"] = self._kwargs_lens_eqn_solver.get(
                "x_center", x_center
            )
            self._kwargs_lens_eqn_solver["y_center"] = self._kwargs_lens_eqn_solver.get(
                "y_center", y_center
            )
        else:
            self._kwargs_lens_eqn_solver["search_window"] = search_window
            self._kwargs_lens_eqn_solver["x_center"] = x_center
            self._kwargs_lens_eqn_solver["y_center"] = y_center

    def update_lens_model(self, lens_model_class):
        """

        :param lens_model_class: instance of LensModel class
        :return: update instance of lens model class
        """
        self.delete_lens_model_cache()
        self._lens_model = lens_model_class
        for model in self._point_source_list:
            model.update_lens_model(lens_model_class=lens_model_class)

    def delete_lens_model_cache(self):
        """Deletes the variables saved for a specific lens model.

        :return: None
        """
        for model in self._point_source_list:
            model.delete_lens_model_cache()

    def set_save_cache(self, save_cache):
        """Set the save cache boolean to new value.

        :param save_cache: bool, if True, saves (or uses a previously saved) values
        :return: updated class and subclass instances to either save or not save the
            point source information in cache
        """
        self._set_save_cache(save_cache)
        self._save_cache = save_cache

    def _set_save_cache(self, save_cache):
        """Set the save cache boolean to new value. This function is for use within this
        class for temporarily set the cache within a single routine.

        :param save_cache: bool, if True, saves (or uses a previously saved) values
        :return: None
        """
        for model in self._point_source_list:
            model.set_save_cache(save_cache)

    def k_list(self, k):
        """

        :param k: index of point source model
        :return: list of lengths of images with corresponding lens models in the frame (or None if not multi-frame)
        """
        if self._index_lens_model_list is not None:
            k_list = []
            for point_source_frame in self._point_source_frame_list[k]:
                k_list.append(self._index_lens_model_list[point_source_frame])
        else:
            k_list = None
        return k_list

    def source_position(self, kwargs_ps, kwargs_lens):
        """Intrinsic source positions of the point sources.

        :param kwargs_ps: keyword argument list of point source models
        :param kwargs_lens: keyword argument list of lens models
        :return: list of source positions for each point source model
        """
        x_source_list = []
        y_source_list = []
        for i, model in enumerate(self._point_source_list):
            kwargs = kwargs_ps[i]
            x_source, y_source = model.source_position(kwargs, kwargs_lens)
            x_source_list.append(x_source)
            y_source_list.append(y_source)
        return x_source_list, y_source_list

    def image_position(
        self,
        kwargs_ps,
        kwargs_lens,
        k=None,
        original_position=False,
        additional_images=False,
    ):
        """Image positions as observed on the sky of the point sources.

        :param kwargs_ps: point source parameter keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param k: None, int or boolean list; only returns a subset of the model
            predictions
        :param original_position: boolean (only applies to 'LENSED_POSITION' models),
            returns the image positions in the model parameters and does not re-compute
            images (which might be differently ordered) in case of the lens equation
            solver
        :param additional_images: if True, solves the lens equation for additional
            images
        :type additional_images: bool
        :return: list of: list of image positions per point source model component
        """
        x_image_list = []
        y_image_list = []
        for i, model in enumerate(self._point_source_list):
            if k is None or k == i:
                kwargs = kwargs_ps[i]
                x_image, y_image = model.image_position(
                    kwargs,
                    kwargs_lens,
                    magnification_limit=self._magnification_limit,
                    kwargs_lens_eqn_solver=self._kwargs_lens_eqn_solver,
                    additional_images=additional_images,
                )
                # this takes action when new images are computed not necessary in order
                if (
                    original_position is True
                    and additional_images is True
                    and self.point_source_type_list[i] == "LENSED_POSITION"
                ):
                    x_o, y_o = kwargs["ra_image"], kwargs["dec_image"]
                    x_image, y_image = _sort_position_by_original(
                        x_o, y_o, x_image, y_image
                    )

                x_image_list.append(x_image)
                y_image_list.append(y_image)
        return x_image_list, y_image_list

    def point_source_list(self, kwargs_ps, kwargs_lens, k=None, with_amp=True):
        """Returns the coordinates and amplitudes of all point sources in a single
        array.

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param k: None, int or list of int's to select a subset of the point source
            models in the return
        :param with_amp: bool, if False, ignores the amplitude parameters in the return
            and instead provides ones for each point source image
        :return: ra_array, dec_array, amp_array
        """
        # here we save the cache of the individual models but do not overwrite the class boolean variable to do so
        self._set_save_cache(True)
        # we make sure we do not re-compute the image positions twice when evaluating position and their amplitudes
        ra_list, dec_list = self.image_position(kwargs_ps, kwargs_lens, k=k)
        if with_amp is True:
            amp_list = self.image_amplitude(kwargs_ps, kwargs_lens, k=k)

        # here we delete the individual modeling caches in case this was the option
        if self._save_cache is False:
            self.delete_lens_model_cache()
            self._set_save_cache(self._save_cache)

        ra_array, dec_array, amp_array = [], [], []
        for i, ra in enumerate(ra_list):
            for j in range(len(ra)):
                # Remove images with zero amplitude so that they do not have to be rendered.
                if with_amp is True and amp_list[i][j] == 0:
                    continue

                ra_array.append(ra_list[i][j])
                dec_array.append(dec_list[i][j])
                if with_amp:
                    amp_array.append(amp_list[i][j])
                else:
                    amp_array.append(1.0)
        return ra_array, dec_array, amp_array

    def num_basis(self, kwargs_ps, kwargs_lens):
        """Number of basis functions for linear inversion.

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :return: int
        """
        n = 0
        ra_pos_list, dec_pos_list = self.image_position(kwargs_ps, kwargs_lens)
        for i, model in enumerate(self.point_source_type_list):
            if self._flux_from_point_source_list[i]:
                if self._fixed_magnification_list[i]:
                    n += 1
                else:
                    n += len(ra_pos_list[i])
        return n

    def image_amplitude(self, kwargs_ps, kwargs_lens, k=None):
        """Returns the image amplitudes.

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param k: None, int or list of int's to select a subset of the point source
            models in the return
        :return: list of image amplitudes per model component
        """
        amp_list = []
        for i, model in enumerate(self._point_source_list):
            if k is None or k == i:
                image_amp = model.image_amplitude(
                    kwargs_ps=kwargs_ps[i],
                    kwargs_lens=kwargs_lens,
                    kwargs_lens_eqn_solver=self._kwargs_lens_eqn_solver,
                )
                if self._flux_from_point_source_list[i]:
                    amp_list.append(image_amp)
                else:
                    amp_list.append(np.zeros_like(image_amp))

        return amp_list

    def source_amplitude(self, kwargs_ps, kwargs_lens):
        """Intrinsic (unlensed) point source amplitudes.

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :return: list of intrinsic (unlensed) point source amplitudes
        """
        amp_list = []
        for i, model in enumerate(self._point_source_list):
            source_amp = model.source_amplitude(
                kwargs_ps=kwargs_ps[i], kwargs_lens=kwargs_lens
            )
            if self._flux_from_point_source_list[i]:
                amp_list.append(source_amp)
            else:
                amp_list.append(np.zeros_like(source_amp))
        return amp_list

    def linear_response_set(self, kwargs_ps, kwargs_lens=None, with_amp=False):
        """

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param with_amp: bool, if True returns the image amplitude derived from kwargs_ps,
         otherwise the magnification of the lens model
        :return: ra_pos, dec_pos, amp, n
        """
        ra_pos = []
        dec_pos = []
        amp = []
        self._set_save_cache(True)
        x_image_list, y_image_list = self.image_position(kwargs_ps, kwargs_lens)
        for i, model in enumerate(self._point_source_list):
            if self._flux_from_point_source_list[i]:
                x_pos = x_image_list[i]
                y_pos = y_image_list[i]
                if self._fixed_magnification_list[i]:
                    ra_pos.append(list(x_pos))
                    dec_pos.append(list(y_pos))
                    if with_amp:
                        mag = self.image_amplitude(kwargs_ps, kwargs_lens, k=i)[0]
                    else:
                        mag = self._lens_model.magnification(x_pos, y_pos, kwargs_lens)
                        mag = np.abs(mag)
                    amp.append(list(mag))
                else:
                    if with_amp:
                        mag = self.image_amplitude(kwargs_ps, kwargs_lens, k=i)[0]
                    else:
                        mag = np.ones_like(x_pos)
                    for j in range(len(x_pos)):
                        ra_pos.append([x_pos[j]])
                        dec_pos.append([y_pos[j]])
                        amp.append([mag[j]])
        n = len(ra_pos)
        if self._save_cache is False:
            self.delete_lens_model_cache()
            self._set_save_cache(self._save_cache)
        return ra_pos, dec_pos, amp, n

    def update_linear(self, param, i, kwargs_ps, kwargs_lens):
        """

        :param param: list of floats corresponding ot the parameters being sampled
        :param i: index of the first parameter relevant for this class
        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :return: kwargs_ps with updated linear parameters, index of the next parameter relevant for another class
        """
        ra_pos_list, dec_pos_list = self.image_position(kwargs_ps, kwargs_lens)
        for k, model in enumerate(self._point_source_list):
            if self._flux_from_point_source_list[k]:
                kwargs = kwargs_ps[k]
                if self._fixed_magnification_list[k]:
                    kwargs["source_amp"] = param[i]
                    i += 1
                else:
                    n_points = len(ra_pos_list[k])
                    kwargs["point_amp"] = np.array(param[i : i + n_points])
                    i += n_points
        return kwargs_ps, i

    def linear_param_from_kwargs(self, kwargs_list):
        """Inverse function of update_linear() returning the linear amplitude list for
        the keyword argument list.

        :param kwargs_list: model parameters including the linear amplitude parameters
        :type kwargs_list: list of keyword arguments
        :return: list of linear amplitude parameters
        :rtype: list
        """
        param = []
        for k, model in enumerate(self._point_source_list):
            if self._flux_from_point_source_list[k]:
                kwargs = kwargs_list[k]
                if self._fixed_magnification_list[k]:
                    param.append(kwargs["source_amp"])
                else:
                    for a in kwargs["point_amp"]:
                        param.append(a)
        return param

    def check_image_positions(self, kwargs_ps, kwargs_lens, tolerance=0.001):
        """Checks whether the point sources in kwargs_ps satisfy the lens equation with
        a tolerance (computed by ray-tracing in the source plane)

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param tolerance: Euclidian distance between the source positions ray-traced
            backwards to be tolerated
        :return: bool: True, if requirement on tolerance is fulfilled, False if not.
        """
        x_image_list, y_image_list = self.image_position(kwargs_ps, kwargs_lens)
        for i, model in enumerate(self.point_source_type_list):
            if model in ["LENSED_POSITION", "SOURCE_POSITION"]:
                x_pos = x_image_list[i]
                y_pos = y_image_list[i]
                # TODO: ray-trace to specific source redshift
                x_source, y_source = self._lens_model.ray_shooting(
                    x_pos, y_pos, kwargs_lens
                )
                dist = np.sqrt(
                    (x_source - x_source[0]) ** 2 + (y_source - y_source[0]) ** 2
                )
                if np.max(dist) > tolerance:
                    return False
        return True

    def set_amplitudes(self, amp_list, kwargs_ps):
        """Translates the amplitude parameters into the convention of the keyword
        argument list currently only used in SimAPI to transform magnitudes to
        amplitudes in the lenstronomy conventions.

        :param amp_list: list of model amplitudes for each point source model. This list
            should include all of the point source models even if flux_from_point_source
            is False for any of them. In that case, the amplitudes will not be changed
            for those models.
        :param kwargs_ps: list of point source keywords
        :return: overwrites kwargs_ps with new amplitudes
        """
        kwargs_list = copy.deepcopy(kwargs_ps)
        for i, model in enumerate(self.point_source_type_list):
            if self._flux_from_point_source_list[i]:
                amp = amp_list[i]
                if model == "UNLENSED":
                    kwargs_list[i]["point_amp"] = amp
                elif model in ["LENSED_POSITION", "SOURCE_POSITION"]:
                    if self._fixed_magnification_list[i] is True:
                        kwargs_list[i]["source_amp"] = amp
                    else:
                        kwargs_list[i]["point_amp"] = amp
        return kwargs_list

    @classmethod
    def check_positive_flux(cls, kwargs_ps):
        """Check whether inferred linear parameters are positive.

        :param kwargs_ps: point source keyword argument list
        :return: bool, True, if all 'point_amp' parameters are positive semi-definite
        """
        pos_bool = True
        for kwargs in kwargs_ps:
            if "point_amp" in kwargs:
                point_amp = np.asarray(kwargs["point_amp"])
                if not np.all(point_amp >= 0):
                    pos_bool = False
                    break
            if "source_amp" in kwargs:
                point_amp = np.asarray(kwargs["source_amp"])
                if not np.all(point_amp >= 0):
                    pos_bool = False
                    break
        return pos_bool


def _sort_position_by_original(x_o, y_o, x_solved, y_solved):
    """Sorting new image positions such that the old order is best preserved.

    :param x_o: numpy array; original image positions
    :param y_o: numpy array; original image positions
    :param x_solved: numpy array; solved image positions with potentially more or fewer
        images
    :param y_solved: numpy array; solved image positions with potentially more or fewer
        images
    :return: sorted new image positions with the order best matching the original
        positions first, and then all other images in the same order as solved for
    """
    if len(x_o) > len(x_solved):
        # if new images are less , then return the original images (no sorting required)
        x_solved_new, y_solved_new = x_o, y_o
    else:
        x_solved_new, y_solved_new = [], []
        for i in range(len(x_o)):
            x, y = x_o[i], y_o[i]
            r2_i = (x - x_solved) ** 2 + (y - y_solved) ** 2
            # index of minimum radios
            index = np.argmin(r2_i)
            x_solved_new.append(x_solved[index])
            y_solved_new.append(y_solved[index])
            # delete this index
            x_solved = np.delete(x_solved, index)
            y_solved = np.delete(y_solved, index)
        # now we append the remaining additional images in the same order behind the original ones
        x_solved_new = np.append(np.array(x_solved_new), x_solved)
        y_solved_new = np.append(np.array(y_solved_new), y_solved)
    return x_solved_new, y_solved_new
