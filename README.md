# PhaseGUI Application Manual

## Overview
**PhaseGUI** is a graphical application for visualizing and processing grayscale PNG images using various darkfield and phase-based techniques. It supports single or batch processing for two channels, with flexible parameter control and result saving.

---

## 1. Input Section
- **Channel 1/2 Path:**  
  - Click the `...` button to select a PNG file or a directory for each channel.
  - The path will appear in the corresponding field.
  - Path to channel 2 will be selected automatically after selection of channel 1, if possible.
  - When you select a file, the save path is set automatically to a parallel `phase` folder.

- **Load single:**  
  - Loads the selected image for each channel and displays it in the preview area.

- **Load all and compute:**  
  - Loads all PNG images in the selected directories for both channels.
  - Automatically runs the selected technique on all images.

---

## 2. Technique and Parameters
- **Technique:**  
  - Choose from:  
    - Darkfield  
    - Phase-Shifted Darkfield  
    - Intensity Weighted Darkfield  
    - Spiral Darkfield  
  - The parameter panel below updates to match the selected technique.

- **Parameters:**  
  - Adjust technique-specific parameters (e.g., `rho1`, `rho2`, `phase shift`, `alpha`, `m`).
  - `rho2` is automatically constrained to half the smallest image dimension.

- **Reference Channel:**  
  - Select which channel is used as the reference for display. (For currently implemented methods, the reference selection does not apply.)
  - Only available channels can be selected.

- **Compute:**  
  - Runs the selected technique on all loaded images for both channels.
  - Results are shown in the output panels.

---

## 3. Output and Preview
- **Output 1/2:**  
  - Show the processed result for channel 1 and channel 2, respectively.

- **Preview:**  
  - Shows the mask used for the most recent computation.

---

## 4. Saving Results
- **Save Path 1/2:**  
  - Automatically set to a `phase` folder parallel to the input directory for each channel.
  - You can change it manually if needed.
  - If both channel folders are in the same directory, make sure to change save path to avoid overwriting. 

- **Save:**  
  - Saves all computed results for each channel in a subfolder named after the technique (e.g., `phase/darkfield/`).

---

## 5. General Notes
- All images must be 8-bit grayscale PNGs and of the same size.
- The application disables controls that are not valid for the current state (e.g., reference channel selection if only one channel is loaded).
- All computations and displays update automatically when you change technique, parameters, or reference.

---

**Tip:**  
Hover over buttons for tooltips. For best results, organize your input images in separate folders for each channel.

---

**Troubleshooting:**  
- If the compute button is disabled, make sure at least one image is loaded.
- If you see unexpected results, check that your images are valid PNGs and have the same dimensions.

---

**Enjoy exploring phase and darkfield image processing!**
