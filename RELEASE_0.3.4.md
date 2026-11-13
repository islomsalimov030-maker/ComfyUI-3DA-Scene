# 0.3.4 — result in the panel megapixels; the prop no longer disappears off-frame

- **Final photo size**: `3DA · final room + rigid props` composes at the H3 raster (the panel megapixels).
  Before, the result was resized to the 1344×768 room preview (≈1 MP) even with 2 MP on the panel.
- **Prop off-frame**: a prop placed by anchor offsets (`3DA Prop`: anchor + x/y/z) that lands outside the camera
  frame is moved toward its anchor until it is inside; the status says so. Props moved in the 3D window are untouched.
- **3D window**: warning «Предмет …: вне кадра — выберите его в списке «Объект» и подвиньте в кадр».
- Recommended `3DA Prop` default: anchor `char_hand_r`, x/y/z = 0 (prop in the right hand).
