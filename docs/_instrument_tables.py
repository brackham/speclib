"""Render registry-backed reference tables during the offline Sphinx build."""
from pathlib import Path

from speclib import get_profile, list_instrument_profiles


def write_tables(directory):
    output = Path(directory) / '_build' / 'instrument_tables'
    output.mkdir(parents=True, exist_ok=True)
    modes = ['.. list-table:: NIRSpec BOTS/S1600A1/SUB2048 (vacuum micrometers)',
             '   :header-rows: 1', '',
             '   * - Profile identifier', '     - Valid intervals',
             '     - Bundled curve', '     - Original data', '     - Coverage source']
    sources = ['.. list-table:: Original STScI dispersion tables',
               '   :header-rows: 1', '',
               '   * - Bundled curve', '     - Original filename',
               '     - FITS creation timestamp (UTC)', '     - Rows']
    seen = set()
    for mode in list_instrument_profiles():
        if not mode.startswith('jwst/nirspec/'):
            continue
        profile = get_profile(mode)
        record = profile.provenance
        coverage = '; '.join(f'{a:g}–{b:g}' for a, b in profile.valid_intervals.value)
        modes += [f'   * - ``{mode}``', f'     - {coverage}',
                  f'     - ``{record["output_file"]}``',
                  f'     - `FITS <{record["url"]}>`__',
                  f'     - `BOTS Table 1 <{profile.coverage_source_url}>`__']
        if record['output_file'] not in seen:
            seen.add(record['output_file'])
            sources += [f'   * - ``{record["output_file"]}``',
                        f'     - `{record["filename"]} <{record["url"]}>`__',
                        f'     - {record["fits_date"]}', f'     - {record["rows"]}']
    (output/'modes.rst').write_text('\n'.join(modes)+'\n')
    (output/'sources.rst').write_text('\n'.join(sources)+'\n')
    _write_phase2_tables(output)
    _write_nircam_tables(output)


def _write_phase2_tables(output):
    modes = ['.. list-table:: STIS, WFC3 and SOSS (vacuum micrometers)',
             '   :header-rows: 1', '', '   * - Profile identifier',
             '     - Configuration', '     - Valid interval', '     - Quality',
             '     - Coverage authority']
    sources = ['.. list-table:: Phase 2 resolution prescriptions',
               '   :header-rows: 1', '', '   * - Bundled curve',
               '     - Resolution source', '     - Adopted prescription']
    widths = ['.. list-table:: STIS model LSF half-height checks (52X2)',
              '   :header-rows: 1', '', '   * - Original LSF',
              '     - Wavelength (µm)', '     - Derived FWHM (pixels)',
              '     - Handbook FWHM (pixels)', '     - Width difference (%)']
    for mode in list_instrument_profiles():
        profile = get_profile(mode)
        if not profile.provenance_ref.startswith('phase2_sources.json#'):
            continue
        record = profile.provenance
        coverage = '; '.join(f'{a:g}–{b:g}' for a, b in profile.valid_intervals.value)
        modes += [f'   * - ``{mode}``', f'     - {profile.configuration}',
                  f'     - {coverage}', f'     - {profile.quality}',
                  f'     - `Reference <{profile.coverage_source_url}>`__']
        if 'lsf_sources' in record:
            links = '; '.join(f'`{s["filename"]} <{s["url"]}>`__'
                              for s in record['lsf_sources'])
            anchors = '; '.join(f'{w:g} µm: {p:g} pixels'
                                for w, p in record['handbook_anchors_um_pixels'])
            prescription = (f'LSF half-height width × {record["dispersion_angstrom_per_pixel"]:g}'
                            f' Å/pixel (`dispersion <{record["dispersion_source_url"]}>`__)')
            if anchors:
                prescription += f'; additional `handbook anchors <{record["handbook_source_url"]}>`__: {anchors}'
            checks = dict(record['handbook_comparison_um_pixels'])
            for source, (wave, width) in zip(record['lsf_sources'], record['derived_lsf_anchors_um_pixels']):
                widths += [f'   * - `{source["filename"]} <{source["url"]}>`__',
                           f'     - {wave:g}', f'     - {width:.6f}',
                           f'     - `{checks[wave]:g} <{record["handbook_source_url"]}>`__',
                           f'     - {100 * (width / checks[wave] - 1):+.3f}']
        else:
            c = record['coefficients']
            if 'reference_wavelength_um' in c:
                links = f'`WFC3 Table 8.1 <{record["documentation_url"]}>`__'
                prescription = (f'R = {c["resolving_power"]:g} × λ / {c["reference_wavelength_um"]:g} µm; '
                                f'reference dispersion {c["dispersion_nm_per_pixel"]:g} nm/pixel')
            else:
                links = (f'`Albert et al., Table 2 / §5.3 <{record["publication_url"]}>`__ '
                         f'(`preprint <{record["preprint_url"]}>`__)')
                prescription = (f'R = λ / ({c["resolution_pixels"]:g} × '
                                f'{c["dispersion_nm_per_pixel"]:g} nm); order {c["order"]}')
        sources += [f'   * - ``{record["output_file"]}``',
                    f'     - {links}', f'     - {prescription}']
    for name, lines in [('phase2_modes', modes), ('phase2_sources', sources), ('stis_widths', widths)]:
        (output/f'{name}.rst').write_text('\n'.join(lines)+'\n')


def _write_nircam_tables(output):
    modes = ['.. list-table:: NIRCam Module A Grism R time series (vacuum micrometers)',
             '   :header-rows: 1', '', '   * - Canonical profile identifier',
             '     - Filter', '     - Filter 5% interval',
             '     - Speclib output interval', '     - Coverage source']
    sources = ['.. list-table:: NIRCam nominal resolving-power prescription',
               '   :header-rows: 1', '', '   * - Shared curve',
               '     - Original prescription', '     - Wavelength support']
    seen = set()
    for mode in list_instrument_profiles():
        if not mode.startswith('jwst/nircam/'):
            continue
        p = get_profile(mode)
        record = p.provenance
        coverage = '; '.join(f'{a:g}–{b:g}' for a, b in p.valid_intervals.value)
        name = mode.split('/')[-1]
        a, b = record['filter_5_percent_limits_um'][name]
        modes += [f'   * - ``{mode}``', f'     - {name.upper()}',
                  f'     - {a:g}–{b:g}', f'     - {coverage}',
                  f'     - `STScI coverage <{p.coverage_source_url}>`__']
        if record['output_file'] not in seen:
            seen.add(record['output_file'])
            lo, hi = record['calibration_domain_um']
            sources += [f'   * - ``{record["output_file"]}``',
                        f'     - `Greene et al. (2017), §3.2 / Figure 2 <{record["publication_url"]}>`__ '
                        f'(`original paper <{record["original_publication_url"]}>`__); '
                        f'`STScI Figure 1 <{record["documentation_url"]}>`__',
                        f'     - {lo:g}–{hi:g} µm (see boundary assumption below)']
    (output/'nircam_modes.rst').write_text('\n'.join(modes)+'\n')
    (output/'nircam_sources.rst').write_text('\n'.join(sources)+'\n')
