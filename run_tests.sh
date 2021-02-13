#!/usr/bin/env bash
# echo on
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
