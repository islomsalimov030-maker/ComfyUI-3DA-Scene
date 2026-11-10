# 0.1.6 candidate

Based on 0.1.5; its SAM input dtype correction is retained.

- Automatically generated rooms use the reconstructed guide as their scale reference when no manual prompt context exists.
- Room assets resolve beside the bound profile when no explicit mapping exists. Server profiles and assets remain constrained to the data root. Camera validation and plate fingerprints remain enabled.
- Ambiguous fragmented floor components can be grouped as an observed coplanar union within the existing 2 cm median height range. No holes are filled, no stair chaining occurs, and largest/lowest plane disagreement still fails.
- Location and pose results record cache flags and stage timings for cold-run evidence.

No new nodes, dependencies or models. Existing geometry/H3 validation gates remain enabled. Diagnostic export permits editing an existing rejected mesh without accepting it for H3.

Server acceptance is pending installation and an explicitly approved restart. This candidate is not evidence of a completed editor-to-final run.
