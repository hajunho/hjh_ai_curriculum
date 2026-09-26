# Lecture 09 · Level 00 — How Computers See Images

> See for yourself — by printing a shape image as text art — that inside a computer, a photo is nothing but a grid of numbers.
**Difficulty** ⭐ / **Prerequisites** none (the NumPy basics from lecture03 level02 help) / **Estimated time** 25 min

## 1. Why This Matters — the Business View

When you hear "the AI looks at a photo and catches defects", it is easy to imagine the computer 'seeing' the way a person does. But to a computer, an image is a **table of numbers** from start to finish. Miss this fact and you will make strange calls on a vision-AI project: "Why would we lower the resolution? Isn't higher quality better?" (the numbers quadruple, and so does the computation), or "The lighting can be a bit dim, right?" (every number shrinks across the board and the model gets confused). To answer questions like these on your own, you need the starting point that an image is numbers.

If you are comfortable in Excel, you actually have an advantage. A single image is exactly **a spreadsheet with a brightness number written in every cell**. By the end of this level you will see that "image processing" is just "spreadsheet math", and that intuition is the foundation for all of lecture09.

## 2. An Analogy

Picture a **mosaic mural**. From a distance it is a human face; up close it is just thousands of small single-color tiles arranged in a grid. The face is not inside any tile. It lives **in the arrangement of the tiles**.

A digital image is exactly this. One tile is a **pixel**, the tile's color is the **pixel value (a number)**, and the number of tiles is the **resolution**. A 16×16 image is a 256-tile mosaic; a smartphone photo (4000×3000) is a 12-million-tile mosaic.

Another analogy is a **seating chart**. Just as you find a seat in a concert hall by "row 3, seat 7", you pick out one pixel of an image with `img[3, 7]`. Row first, column second — the same order as "row, column" in Excel. That is all you need to remember.

## 3. Core Concepts

### 3.1 A grayscale image = a 2-D array of numbers

A grayscale image is a 2-D array shaped `height × width`. The number in each cell is a brightness.

- 0.0 = pure black, 1.0 = pure white, 0.5 = middle gray (sometimes expressed as integers 0–255 instead)
- The shape images in this lecture are 16×16 float arrays with values 0.0–1.0.

In other words, the "picture" on the left below and the "number table" on the right are exactly the same thing.

```
■ □        1.0  0.0
□ ■        0.0  1.0
```

### 3.2 Resolution — the tile-count trade-off

Higher resolution means more detail, but the number count explodes in proportion to width×height. 16×16 is 256 numbers, 32×32 is 1,024 (4×), and 1024×1024 is over a million. In deep learning every input number costs computation, so in practice finding "the minimum resolution that can still solve the task" is an important design decision. If a scratch shows up as 3 pixels at some resolution, that resolution is enough for defect inspection — anything beyond it is pure cost.

### 3.3 Brightness — the size of the numbers

An image being "bright" or "dark" is simply the numbers being large or small. Dim the lighting by half and every pixel value drops by roughly half. The same shape under different lighting becomes a very different array of numbers, which is why "fixing the shooting conditions" accounts for half of data quality when building a vision system. (Adjusting brightness in code is covered in level01.)

### 3.4 What about color? — a preview

A color image is three brightness tables — red, green, blue (RGB) — stacked together: an array shaped `height × width × 3`. This level sticks to grayscale; channels get their full treatment in level01.

### 3.5 Today's data — shapes we draw ourselves

This lecture never downloads a famous dataset. We use 16×16 grayscale shapes (squares, circles, triangles — 3 classes) that `shape_images()` in `common/hjh_data.py` draws on the spot with numpy. Positions and sizes are randomized and noise is mixed in, so despite its size it has all the properties of a "real classification problem".

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

This program shows that a shape image is "the same data" in three representations — a number table, text art, and a PNG.

- **[1]** Creates shape images with `shape_images()` and prints the array's shape, value range, and class composition.
- **[2]** Prints one image as a raw **grid of decimal numbers**. Amid a screenful of values between 0.0 and 1.0, see if you can spot the outline of the shape.
- **[3]** Prints the same image as **text art**. Assign characters like ` .:-=#@` to brightness bands and the number grid suddenly reads as a "picture". We changed nothing but one number→character rule, and now a human eye sees the shape — this is a hands-on demo of "an image is numbers".
- **[4]** The resolution analogy: the 16×16 image gets crushed down to 8×8 and 4×4 (averaging 2×2 blocks) and compared as text art. Watch for the moment where, with fewer tiles, a circle and a square become indistinguishable.
- **[5]** Saves a grid of 9 shape samples per class to `outputs/shapes_grid.png`. Open the file — it is the same array you have been looking at as text.

The heart of the code is the `to_ascii()` function. One line, `int(v * (len(chars) - 1))`, turns a brightness 0–1 into a character index. It is a five-line function showing that rendering is ultimately just "a rule for turning numbers into readable symbols".

## 5. Try It Yourself

1. **(Easy)** Reverse the character palette `" .:-=+*#@"` in `[3]` and run it again. The shape appears inverted (a negative). Explain in one sentence why. (Hint: which character gets assigned to dark pixels now?)
2. **(Medium)** In `[4]`, change the downscaling to use the **maximum** of each block instead of the mean. The shape survives thicker. The name of this method is "max pooling", which you will meet in level04. (Hint: `block.mean()` → `block.max()`)
3. **(Challenge)** Raise the resolution with `shape_images(size=32)` and print the text art. If your terminal is narrow, lines will wrap. Also think about why printing 2 characters per pixel (`chars[idx] * 2`) makes the aspect ratio look natural. (Hint: terminal characters are taller than they are wide)

## 6. Common Mistakes

- **Assuming `img[x, y]`**: NumPy images are `img[row, col]`, i.e. `img[y, x]`. Put the horizontal coordinate first and you will hit bugs where the image seems rotated 90 degrees.
- **Mixing up value ranges**: combine a 0–1 float image with a 0–255 integer image and the screen comes out all white or all black. When you receive an array, make it a habit to print `min/max` first.
- **Believing resolution = quality**: resolution is cost. Using only as much as the task demands is standard practice.
- **Thinking "the computer knows it's a shape"**: at this stage the computer has nothing but an array of 0.0s and 1.0s. Extracting the meaning "shape" out of a number array is what the remaining 11 levels of this lecture are about.

## Next Level Preview

If an image is numbers, then image editing is arithmetic. Raising brightness = addition, raising contrast = multiplication, inverting = subtraction. In level01 we do image operations directly in numpy, including RGB color channels.
