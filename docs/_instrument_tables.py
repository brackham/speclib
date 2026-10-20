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
