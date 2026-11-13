# 0.3.13 — no late-step mosaic with clean shadows

- With `clean_shadows` on, «3DA · native H3 guides + gate» outputs a solid sampling mask: 1.0 over the whole H3 area,
  ~6 px soft edge. Before, fractional strengths (band tail, 0.7 shadow zone) under DifferentialDiffusion with shift 12
  were released only at the last step and got one coarse step → mosaic appearing at the very end of sampling.
- The room outside the person still comes from the photo × H3 darkening (clean shadows); the edge from the soft composite.
- With `clean_shadows` off nothing changes.
