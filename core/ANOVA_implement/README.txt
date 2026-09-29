## Introduction

In the tensor setting, ANOVA is equivalent to cluster expansion: each coordinate splits into the mean part `span{1}` and its orthogonal complement, 
and each ANOVA term is the tensor product of these orthogonal projections. (Takemura 1983)

This viewpoint is also computationally useful: by representing `span{1}` and `span{1}^\perp` in the Fourier basis, we can implement the projection 
with FFT instead of explicit Kronecker constructions, leading to a faster algorithm.

We use the name **cluster expansion** to emphasize this equivalence with ANOVA.