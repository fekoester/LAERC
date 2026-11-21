| model | trainable params (all) | non-embedding | best val loss | val perplexity | bits/char |
|---|---|---|---|---|---|
| FFN only (no context) | 153,508 | 104,356 | 4.172 | 64.9 | 2.496 |
| LAERC v0.1 scaling (ρ≈0.47) | 201,512 | 152,360 | 3.628 | 37.6 | 2.171 |
| LAERC, ρ = 0.9 | 201,512 | 152,360 | 3.582 | 35.9 | 2.143 |
| LAERC, ρ = 0.99 | 201,512 | 152,360 | 3.583 | 36.0 | 2.144 |
| LAERC, multi-timescale (ρ = 0.9, leak 1e-3…1) | 201,512 | 152,360 | 3.491 | 32.8 | 2.089 |
| GPT (small Transformer) | 168,480 | 119,328 | 3.568 | 35.5 | 2.135 |
