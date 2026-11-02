# RESOLVED — this blocker no longer applies

The permuted-label null has been computed on the paper's own activations. See
[RESOLVED.md](permutation_null.md).

The tensors this document said were missing were regenerated from the in-repo processed
split (260 harmful / 6000 harmless) and reproduce the published per-layer RCS to 2.0e-04.
Neither SORRY-Bench access nor a HuggingFace token was required; see
`data_source.md` for why.

One warning for anyone re-running the extraction: on Turing GPUs the repository's default
attention backend silently corrupts the forward pass. See `sdpa_turing_superseded.md`.
