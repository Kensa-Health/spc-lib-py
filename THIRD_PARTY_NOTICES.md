# Third-party notices

## NHSRplotthedots

Parts of this library follow, and in places are ported from, the R package
[NHSRplotthedots](https://github.com/nhs-r-community/NHSRplotthedots) (version 0.2.2 was
compared), which implements the NHS "Making Data Count" XmR chart methodology and is
distributed under the MIT licence below. The affected areas are:

- the XmR limit calculation and outlier screening (`spc/core/limits.py`:
  `compute_limit_stats`, the `2.66`, `2 * (2.66 / 3)` and `3.267` constants and the
  `fix_y` / `mean_col` / `amr` structure);
- the special-cause rules and their labels (`spc/core/rules.py`: the seven-point trend,
  the seven-point shift, two-in-three near the limit, and the classification of a flagged
  point as improvement, concern or neutral);
- the assurance calculation (`spc/core/limits.py`: `assurance_type`);
- rebase grouping (`spc/core/segments.py`);
- the names of several options and result fields and values (`value_field`, `date_field`,
  `facet_field`, `rebase`, `target`, `improvement_direction`, `fix_after_n_points`,
  `screen_outliers`, `point_type`, `assurance_type`, `outside_limits`, `close_to_limits`,
  `relative_to_mean`, and values such as `special_cause_concern`).

The rest of the library (the attribute, Xbar, rare-event, Laney, CUSUM and Pareto charts,
facets, validation and rendering) is original to this project.

```text
MIT License

Copyright (c) 2021 NHS-R Community

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
