# 0.3.16 — rooms without detected support surfaces

- In a room where no support surface was found, the fallback placement had no BODY-25 keypoints and the scene render
  failed with «KeyError: 'px_size'»: the 3D window kept the previous scene. The fallback now writes the keypoints, and
  the renderer no longer requires them. The character is placed in the frame as a warning; adjust it in the 3D window.
