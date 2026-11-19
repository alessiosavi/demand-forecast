# Legacy v5 export fixture

A real export written by the code at revision `886d86e`, before the `derive` switch existed. Tests use it to
show that an export without that key still replays its parity cases and scores unchanged.

It cannot be regenerated from the current tree. It was produced once with that revision's export writer from
the smallest test model (seed 123): one category case of the stock parity set shortened to one context
patch, seven minimal cases (context 16, horizon 2), and `forecasts.json`, a forecast scored by that
revision's adapter with the variant `full`.

Files: `config.json`, `model.safetensors`, `parity.npz`, `forecasts.json` (about 295 kB together). Do not
rewrite them with newer code: the point of the fixture is that newer code did not write it.
