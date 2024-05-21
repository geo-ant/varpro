# ToDo List

[ ] Better interface for fit result: successful fits should not have optional
linear coefficients. Use const generic bool like I did for the MRHS case?
[ ] Fit statistics (and confidence bands, but also correlation matrix etc) for
problems with multiple RHS
[ ] Provide a more convenient way to add fittable offsets (plus some safeguards, such
that offsets cannot be added twice). Also think of a better name than fittable offset,
but make it clear that it is not just a constant offset, but
one that comes with an extra parameter.
