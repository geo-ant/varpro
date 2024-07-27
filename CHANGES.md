# Changes for `varpro`

This is the changelog for the `varpro` library.
See also here for a [version history](https://crates.io/crates/varpro/versions).

## 0.10.0

- Updated dependencies to current versions.
- Updates to documentation

## 0.9.0

- We can now calculate confidence bands using via `FitStatistics`. Only available
for problems with a single RHS, though. 
- The API for calculating the correlation matrix changed to on-the-fly
calculations, deprecated (a drop-in replacement) for the old API.
- Cleaner separation (API changes) for problems with single and multiple right
hand sides.
- Removed some overly clever (but ultimately bad) generics from the `SeparableModelBuilder`
that were making it hard to use to add functions in a loop. State machine is now
a runtime thing.

## 0.8.0 Multiple Right Hand Sides

- Observations can now be a matrix, in this case a global fit with multiple right
hand sides is performed.
- Linear coefficients are now in matrix form. For a single right hand side, this
is still a matrix with one column, for multiple right hand sides these are now
the coefficient vectors (columns) corresponding to the respective right hand
side column of the observation matrix.
- Removed lots of generics internally.
- Deprecated the new function of the levmar solver.

## 0.7.0 Fit Statistics

- Deprecate the `minimize` function of the LevMarSolver and
replace it with `fit`, with a slightly different API.
- Add a function `fit_with_statistics` and a `statistics` module
that calculates additional statistical information about the fit 
(if successful). It allows us to extract the estimated standard
errors of the model parameters (both linear and nonlinear) but also
provides more complete information like the covariance matrix and
the correlation matrix.
- add more exhaustive tests 
- add original varpro matlab code from DP O'Leary and BW Rust
with permission
- bump benchmarking dependencies

## 0.6.0 Performance Improvements and Benchmarking

- Add benchtests
- Change solver API to depend on separable model trait instead of concrete impl,
which allows us to optimize our models for performance
- changes in model builder api, see documentation for new usage

## 0.5.0 nalgebra Update

- Upgrade nalgebra and levenberg_marquardt dependencies to current versions
- Fix deprecation warnings and expand test coverage slightly

## 0.4.1 Misc Maintanance

- Remove snafu dependency and replace by `thiserror`
- Use uninitialized matrices instead of zero initialisation for places where contents will be overwritten anyways
- Fix new clippy lints
- Redo the code coverage workflow and switch to coveralls from codecov

## 0.3.0, 0.4.0

Upgrade dependencies


## 0.2.0

- Update `levenberg_marquardt` dependency to 0.8.0
- Update `nalgebra` dependency to 0.25.3

## 0.1.0

Initial release
