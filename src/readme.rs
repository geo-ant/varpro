// this is a way of running doctests on my readme so I know the code is still valid
// https://github.com/rust-lang/cargo/issues/383.
// That means errors that occur here need to be fixed in the readme.
#[doc = include_str!("../README.md")]
struct _Readme {}
