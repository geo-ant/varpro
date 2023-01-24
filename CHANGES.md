# Changes for `varpro`

This is the changelog for the `varpro` library. See also here for a [version history](https://crates.io/crates/varpro/versions).

## 0.4.1
- Remove snafu dependency and replace by `thiserror`
- Use uninitialized matrices instead of zero initialisation for places where contents will be overwritten anyways
- Fix new clippy lints

## 0.3.0, 0.4.0
Upgrade dependencies


## 0.2.0

- Update `levenberg_marquardt` dependency to 0.8.0
- Update `nalgebra` dependency to 0.25.3

## 0.1.0
Initial release
