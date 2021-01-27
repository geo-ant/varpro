extern crate nalgebra;
extern crate snafu;
extern crate levenberg_marquardt;

pub mod model;
pub mod types;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
