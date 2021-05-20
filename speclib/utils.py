import astropy.units as u

__all__ = ["vac2air", "air2vac"]


@u.quantity_input(wl_vac=u.AA)
def vac2air(wl_vac):
    """
    Convert vacuum to air wavelength.

    See http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion.

    Parameters
    ----------
    wl_vac : `~astropy.units.Quantity`
        Vacuum wavelength.

    Returns
    -------
    wl_air : `~astropy.units.Quantity`
        Air wavelength.

    See Also
    --------
    air2vac
        Convert air to vacuum wavelength.
    """
    # Wavelengths must be specified in Å
    orig_unit = wl_vac.unit
    wl_vac = wl_vac.to(u.AA).value

    s = 10.0 ** 4 / wl_vac
    n = 1 + 0.0000834254 + 0.02406147 / (130.0 - s ** 2) + 0.00015998 / (38.9 - s ** 2)
    wl_air = wl_vac / n

    # Convert back to original unit, if necessary
    wl_air = (wl_air * u.AA).to(orig_unit)

    return wl_air


@u.quantity_input(wl_air=u.AA)
def air2vac(wl_air):
    """
    Convert air to vacuum wavelength.

    See http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion.

    Parameters
    ----------
     wl_air : `~astropy.units.Quantity`
        Air wavelength.

    Returns
    -------
    wl_vac : `~astropy.units.Quantity`
        Vacuum wavelength.

    See Also
    --------
    vac2air
        Convert vacuum to air wavelength.
    """
    # Wavelengths must be specified in Å
    orig_unit = wl_air.unit
    wl_air = wl_air.to(u.AA).value

    s = 10.0 ** 4 / wl_air
    n = (
        1
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s ** 2)
        + 0.0001599740894897 / (38.92568793293 - s ** 2)
    )
    wl_vac = wl_air * n

    # Convert back to original unit, if necessary
    wl_vac = (wl_vac * u.AA).to(orig_unit)

    return wl_vac
