import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from scipy.stats import median_abs_deviation


def make_offset_plot(dra, ddec, matched, fig=None):
    """
    Plot the results of the matching process.

    Args:
        dra (astropy.units.Quanity): The offsets measured along the RA axis.
        ddec (astropy.units.Quanity): The offsets measured along the Dec axis.
        matched (ndarray): A flag indicating which sources are considered matched.
        fig (matplotlib.pyplot.figure): A canvas used to create the plot.
    """
    if fig is None:
        fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    dx = dra.to(u.arcsec).value
    dy = ddec.to(u.arcsec).value

    median_x = np.median(dx[matched])
    median_y = np.median(dy[matched])

    mad_x = median_abs_deviation(dx[matched])
    mad_y = median_abs_deviation(dy[matched])

    ax.scatter(dx[matched], dy[matched], marker='d', s=10, c='C1', alpha=0.4, zorder=2)
    ax.scatter(dx[~matched], dy[~matched], marker='+', s=10, c='C3', alpha=0.1, zorder=2)
    ax.plot(0, 0, marker='+', ms=15, mew=2.5, mec='k', zorder=3)

    ell = Ellipse(xy=(median_x, median_y), width=2. * mad_x, height=2. * mad_y, linewidth=1.5, edgecolor='k',
                  facecolor='None', zorder=4)
    ax.add_patch(ell)

    ax.set_xlabel(r'$\Delta x/{\rm arcsec}$', fontsize=16)
    ax.set_ylabel(r'$\Delta y/{\rm arcsec}$', fontsize=16)

    fig.tight_layout()


class MatchCatalogs(object):
    """
    A tool to match two catalogs with each other, after accounting
    for a global or spatially varying offset between them.

    By default, no offset is determined, hence it is assumed that both
    catalogs are in the same frame of reference. In order to determine
    a global offset along RA and Dec, a set of calibration sources must
    be provided for either catalog. To account for a spatially varying
    offset (e.g., in cases where one catalog suffers from distortions),
    the number of nearest neighbors from the calibration set used to
    determine the offset at the position of each source must be provided.

    Args:
        reference (astropy.table.Table): The reference catalogue.
        ra_column (str): The name of the column containing the RA coordinates.
        dec_column (str): The name of the column containing the Dec coordinates.
        calibration (ndarray): A flag indicating which sources should be used to calibrate a global offset.
        mag_column (str): Optional name of column containing magnitude used during calibration.
    """

    def __init__(self, reference: Table, ra_column: str = 'ra', dec_column: str = 'dec',
                 calibration: np.ndarray = None, mag_column: str = None):
        """
        Initialize a new instance of MatchCatalogs.
        """
        assert ra_column in reference.colnames and dec_column in reference.colnames

        self.reference = SkyCoord(ra=u.Quantity(reference[ra_column], u.deg),
                                  dec=u.Quantity(reference[dec_column], u.deg))

        if calibration is not None:
            self.calibration = self.reference[calibration]
        else:
            self.calibration = self.reference

        if mag_column is not None:
            assert mag_column in reference.colnames
            self.reference_mags = reference[mag_column]
        else:
            mag_column = None

    def __call__(self, catalog: Table, ra_column: str = 'ra', dec_column: str = 'dec',
                 max_off: u.Quantity = None, calibration: np.ndarray = None, mag_column: str = None,
                 max_off_calibration: u.Quantity = None, max_off_mag: float = 0.2, neighbors: int = None):
        """
        Compare the provided catalogue to the reference and find the nearest
        match in the latter to each source in the former.

        If requested, a global offset between the catalogs will be measured
        and accounted for prior to the matching. The idea is to use a bright
        subset of the sources in both catalogs, so that the density of sources
        is low enough that despite the offset, the correct counterparts are
        still identified.

        Args:
            catalog (astropy.table.Table): The catalog to be matched to the reference.
            ra_column (str): The name of the column containing the RA coordinates.
            dec_column (str): The name of the column containing the Dec coordinates.
            max_off (astropy.units.Quantity): The maximum offset allowed between matched sources.
            calibration (ndarray): A flag indicating which sources should be used to estimate a global offset.
            mag_column (str): Optional name of column containing magnitude used during calibration.
            max_off_calibration (astropy.units.Quantity): Same as `max_off` for the calibration step.
            max_off_mag (float): The maximum magnitude offset allowed during the calibration step.
            neighbors (int): The number of nearest sources used if local offset calculation is requested.

        Returns:
            matches (ndarray): The matched indices in the catalog and in the reference.
            offset (tuple): The offsets determined in the calibration step.
        """

        assert ra_column in catalog.colnames and dec_column in catalog.colnames

        _sc = SkyCoord(ra=u.Quantity(catalog[ra_column], u.deg),
                       dec=u.Quantity(catalog[dec_column], u.deg), frame='icrs')

        if calibration is not None:
            _calib = calibration.copy()

            idx, d2d, _ = _sc[_calib].match_to_catalog_sky(self.calibration)

            matched = np.ones(idx.shape, dtype=bool)
            if max_off_calibration is not None:
                matched &= d2d < max_off_calibration
            if max_off_mag is not None and mag_column is not None:
                assert mag_column in catalog.colnames
                assert self.reference_mags is not None
                matched &= abs(catalog[mag_column][_calib] - self.reference_mags[idx]) < max_off_mag
            print(matched.sum())
            idx = idx[matched]
            d2d = d2d[matched]

            matches = self.calibration[idx]

            _calib[_calib] = matched
            _dra, _ddec = _sc[_calib].spherical_offsets_to(matches)
            # make_offset_plot(_dra, _ddec, matched[matched])

            if neighbors is None:
                dra_calib = np.median(_dra)
                ddec_calib = np.median(_ddec)
                _sc = _sc.spherical_offsets_by(np.ones(_sc.shape) * dra_calib,
                                               np.ones(_sc.shape) * ddec_calib)
            else:
                dra_calib = Angle(np.zeros(_sc.shape), unit=u.deg)
                ddec_calib = Angle(np.zeros(_sc.shape), unit=u.deg)
                for i in tqdm.tqdm(range(len(_sc))):
                    delta = _sc[i].separation(_sc[_calib])
                    nearest = delta < delta[delta.argsort()[neighbors]]
                    dra_calib[i] = np.median(_dra[nearest])
                    ddec_calib[i] = np.median(_ddec[nearest])
                _sc = _sc.spherical_offsets_by(dra_calib, ddec_calib)
        else:
            dra_calib = None
            ddec_calib = None

        idx, d2d, _ = _sc.match_to_catalog_sky(self.reference)

        matches = self.reference[idx]
        dra, ddec = _sc.spherical_offsets_to(matches)

        if max_off is not None:
            matched = d2d < max_off
            idx = idx[matched]
            d2d = d2d[matched]
        else:
            matched = np.ones(idx.shape, dtype=bool)

        make_offset_plot(dra, ddec, matched=matched)

        return np.vstack([np.flatnonzero(matched), idx]), (dra_calib, ddec_calib)
