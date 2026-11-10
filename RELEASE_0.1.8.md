# 0.1.8 candidate

The 0.1.7 cold run exposed an unsupported half-precision CUDA sparse operation in MHR under the global SAM autocast scope. Server generation was stopped and the full available traceback retained.

- Replace global autocast with temporary input dtype guards on the existing dense refinement modules: prev_to_token_mhr, prev_to_token_mhr_hand and prompt_to_token.
- Match only floating-point dense inputs to their module weight dtype. Keep model weights, hand refinement and the sparse/MHR precision path unchanged.
- Remove every guard on success and on exceptions. The PC path remains unchanged.

Local tests cover fp16/bf16 dense Float promotion, unchanged FP32 sparse math, no autocast leakage and hook cleanup after an exception. Actual server success remains unverified.

No new nodes, dependencies, models or weights. Publication, resuming server tests after STOP and another single idle-queue reboot require explicit approval.
