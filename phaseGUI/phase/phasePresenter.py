from PyQt5 import QtCore, QtGui, QtWidgets
from posixpath import basename, dirname, join
import cv2
from natsort import natsorted
import numpy as np
import os


class PhasePresenter:
    def __init__(self, view, model):
        self.technique_index = 0  
        self.reference_index = 1  
        self.view = view
        self.model = model

    def update_reference_combobox_state(self):
        # Disable reference options for channels that are not loaded
        ch1_loaded = hasattr(self, 'imgs_ch1') and self.imgs_ch1 is not None and self.imgs_ch1.shape[0] > 0
        ch2_loaded = hasattr(self, 'imgs_ch2') and self.imgs_ch2 is not None and self.imgs_ch2.shape[0] > 0
        self.view.referenceComboBox.model().item(0).setEnabled(ch1_loaded)
        self.view.referenceComboBox.model().item(1).setEnabled(ch2_loaded)
        # If the current selection is not enabled, switch to the enabled one
        if not self.view.referenceComboBox.model().item(self.view.referenceComboBox.currentIndex()).isEnabled():
            if ch1_loaded:
                self.view.referenceComboBox.setCurrentIndex(0)
            elif ch2_loaded:
                self.view.referenceComboBox.setCurrentIndex(1)

    def update_compute_button_state(self):
        # Enable compute if at least one image is loaded in either channel
        enabled = (hasattr(self, 'imgs_ch1') and self.imgs_ch1 is not None and self.imgs_ch1.shape[0] > 0) or (hasattr(self, 'imgs_ch2') and self.imgs_ch2 is not None and self.imgs_ch2.shape[0] > 0)
        self.view.computeButton.setEnabled(enabled)
    
    TECHNIQUE_MODEL_MAP = {
        0: 'darkfield',
        1: 'phase_shifted_darkfield',
        2: 'intensity_weighted_darkfield',
        3: 'spiral_darkfield',
    }

    def get_directory(self, directory = False):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if directory:
            dlg.setFileMode(QtWidgets.QFileDialog.Directory)    
        else:
            dlg.setNameFilter("PNG files (*.png)")
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            return filenames[0]
        else: 
            return ''
        
    def set_save_path_for_channel(self, channel):
        """
        Set the save path for the given channel (1 or 2) to a parallel 'phase' folder.
        """
        dirc = getattr(self, f'dirname_ch{channel}', None)
        if dirc:
            parent = dirname(dirc)
            phase_dir = join(parent, 'phase')
            os.makedirs(phase_dir, exist_ok=True)
            getattr(self.view, f'lineEdit_save_path_ch{channel}').setText(phase_dir)
    
    def on_ch1_in_changed(self):
        dirc = self.view.lineEdit_ch1_in.text()
        self.filename_ch1 = basename(dirc)
        self.dirname_ch1 = dirname(dirc)
        self.set_save_path_for_channel(1)
    
    def on_ch2_in_changed(self):
        dirc = self.view.lineEdit_ch2_in.text()
        self.filename_ch2 = basename(dirc)
        self.dirname_ch2 = dirname(dirc)
        self.set_save_path_for_channel(2)

    def on_ch1_in_clicked(self):
        dirc = self.get_directory()
        self.view.lineEdit_ch1_in.setText(dirc)
        self.filename_ch1 = basename(dirc)
        self.dirname_ch1 = dirname(dirc)
        self.set_save_path_for_channel(1)
        try:
            tmp = dirc.replace('channel1', 'channel2')
        except:
            pass
        else:
            self.view.lineEdit_ch2_in.setText(tmp)
            self.filename_ch2 = basename(tmp)
            self.dirname_ch2 = dirname(tmp)
            self.set_save_path_for_channel(2)

    def on_ch2_in_clicked(self):
        dirc = self.get_directory()
        self.view.lineEdit_ch2_in.setText(dirc)
        self.filename_ch2 = basename(dirc)
        self.dirname_ch2 = dirname(dirc)
        self.set_save_path_for_channel(2)

    def on_save_clicked(self):
        # Save results for each channel in phase/<method>/
        technique = self.technique_index
        method_name = self.TECHNIQUE_MODEL_MAP.get(technique, "unknown")
        for ch in [1, 2]:
            imgs = getattr(self, f'results_ch{ch}', None)
            if imgs is not None and imgs.shape[0] > 0:
                save_base = getattr(self.view, f'lineEdit_save_path_ch{ch}').text()
                method_dir = join(save_base, method_name)
                os.makedirs(method_dir, exist_ok=True)
                for i, img in enumerate(imgs):
                    out_path = join(method_dir, f"result_{i:04d}.png")
                    cv2.imwrite(out_path, img)
                print(f"Saved {imgs.shape[0]} images for channel {ch} to {method_dir}")

    def on_save_path_ch1_clicked(self):
        dirc = self.get_directory(directory=True)
        self.view.lineEdit_save_path_ch1.setText(dirc)

    def on_save_path_ch2_clicked(self):
        dirc = self.get_directory(directory=True)
        self.view.lineEdit_save_path_ch2.setText(dirc)

    def clear_image_labels(self):
        self.view.imageLabel_ch1.axes.clear()
        self.view.imageLabel_ch1.draw()
        self.view.imageLabel_ch2.axes.clear()
        self.view.imageLabel_ch2.draw()
        self.view.imageLabel_output1.axes.clear()
        self.view.imageLabel_output1.draw()
        self.view.imageLabel_output2.axes.clear()
        self.view.imageLabel_output2.draw()
        self.view.imageLabel_preview.axes.clear()
        self.view.imageLabel_preview.draw()

    def on_load_single_clicked(self):
        # Load single image for ch1 and ch2, store as ndarrays 
        self.clear_image_labels()
        imgs_ch1, imgs_ch2 = [], []
        if hasattr(self, 'filename_ch1') and hasattr(self, 'dirname_ch1'):
            path_ch1 = os.path.join(self.dirname_ch1, self.filename_ch1)
            img_ch1 = cv2.imread(path_ch1, cv2.IMREAD_GRAYSCALE)
            if img_ch1 is not None:
                imgs_ch1.append(img_ch1)
                print(f"Loaded single image for ch1: {path_ch1}, shape={img_ch1.shape}")
            else:
                print(f"Failed to load image: {path_ch1}")
        if hasattr(self, 'filename_ch2') and hasattr(self, 'dirname_ch2'):
            path_ch2 = os.path.join(self.dirname_ch2, self.filename_ch2)
            img_ch2 = cv2.imread(path_ch2, cv2.IMREAD_GRAYSCALE)
            if img_ch2 is not None:
                imgs_ch2.append(img_ch2)
                print(f"Loaded single image for ch2: {path_ch2}, shape={img_ch2.shape}")
            else:
                print(f"Failed to load image: {path_ch2}")
        # Convert to ndarrays
        self.imgs_ch1 = np.stack(imgs_ch1) if imgs_ch1 else np.empty((0, 0, 0), dtype=np.uint8)
        self.imgs_ch2 = np.stack(imgs_ch2) if imgs_ch2 else np.empty((0, 0, 0), dtype=np.uint8)
        # Display if loaded
        if self.imgs_ch1.shape[0] > 0:
            self.view.imageLabel_ch1.axes.imshow(self.imgs_ch1[0], cmap='gray')
            self.view.imageLabel_ch1.draw()
        if self.imgs_ch2.shape[0] > 0:
            self.view.imageLabel_ch2.axes.imshow(self.imgs_ch2[0], cmap='gray')
            self.view.imageLabel_ch2.draw()
        self.update_compute_button_state()
        self.update_reference_combobox_state()
        self.set_rho2_constraints()


    def on_load_all_and_compute_clicked(self):
        # Load all PNG images in ch1 and ch2 directories as ndarrays
        imgs_ch1, imgs_ch2 = [], []
        if hasattr(self, 'dirname_ch1'):
            files_ch1 = natsorted([f for f in os.listdir(self.dirname_ch1) if f.lower().endswith('.png')])
            for fname in files_ch1:
                path = os.path.join(self.dirname_ch1, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    imgs_ch1.append(img)
            print(f"Loaded {len(imgs_ch1)} images for ch1 from {self.dirname_ch1}")
        if hasattr(self, 'dirname_ch2'):
            files_ch2 = natsorted([f for f in os.listdir(self.dirname_ch2) if f.lower().endswith('.png')])
            for fname in files_ch2:
                path = os.path.join(self.dirname_ch2, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    imgs_ch2.append(img)
            print(f"Loaded {len(imgs_ch2)} images for ch2 from {self.dirname_ch2}")
        # Convert to ndarrays
        self.imgs_ch1 = np.stack(imgs_ch1) if imgs_ch1 else np.empty((0, 0, 0), dtype=np.uint8)
        self.imgs_ch2 = np.stack(imgs_ch2) if imgs_ch2 else np.empty((0, 0, 0), dtype=np.uint8)
        # Optionally display first image
        if self.imgs_ch1.shape[0] > 0:
            self.view.imageLabel_ch1.axes.imshow(self.imgs_ch1[0], cmap='gray')
            self.view.imageLabel_ch1.draw()
        if self.imgs_ch2.shape[0] > 0:
            self.view.imageLabel_ch2.axes.imshow(self.imgs_ch2[0], cmap='gray')
            self.view.imageLabel_ch2.draw()
        self.update_compute_button_state()
        self.update_reference_combobox_state()
        self.set_rho2_constraints()
        self.on_compute_clicked()

    
    def on_compute_clicked(self):
        # Call the computation for the selected technique and reference
        technique = self.technique_index
        imgs_ch1 = self.imgs_ch1 if hasattr(self, 'imgs_ch1') and self.imgs_ch1 is not None and self.imgs_ch1.shape[0] > 0 else None
        imgs_ch2 = self.imgs_ch2 if hasattr(self, 'imgs_ch2') and self.imgs_ch2.shape[0] > 0 else None
        self.results_ch1 = self.masks_ch1 = self.results_ch2 = self.masks_ch2 = None
        # Compute for both channels if available, preallocating arrays
        if imgs_ch1 is not None:
            n, h, w = imgs_ch1.shape
            self.results_ch1 = np.empty((n, h, w), dtype=np.float32)
            self.masks_ch1 = np.empty((n, h, w), dtype=np.uint8)
            if technique == 0:
                rho1 = self.view.spinBox_darkfield_rho1.value()
                rho2 = self.view.spinBox_darkfield_rho2.value()
                for i, img in enumerate(imgs_ch1):
                    out, mask = self.model.darfield(img, rho1, rho2)
                    self.results_ch1[i] = out
                    self.masks_ch1[i] = mask
            elif technique == 1:
                rho1 = self.view.spinBox_psdarkfield_rho1.value()
                rho2 = self.view.spinBox_psdarkfield_rho2.value()
                phaseshift = self.view.spinBox_psdarkfield_phaseshift.value()
                for i, img in enumerate(imgs_ch1):
                    out, mask = self.model.phase_shifted_darkfield(img, rho1, rho2, phaseshift)
                    self.results_ch1[i] = out
                    self.masks_ch1[i] = mask
            elif technique == 2:
                rho1 = self.view.spinBox_iwdarkfield_rho1.value()
                rho2 = self.view.spinBox_iwdarkfield_rho2.value()
                alpha = self.view.spinBox_iwdarkfield_alpha.value()
                for i, img in enumerate(imgs_ch1):
                    out, mask = self.model.intensity_weighted_darkfield(img, rho1, rho2, alpha)
                    self.results_ch1[i] = out
                    self.masks_ch1[i] = mask
            elif technique == 3:
                rho1 = self.view.spinBox_sdarkfield_rho1.value()
                rho2 = self.view.spinBox_sdarkfield_rho2.value()
                m = self.view.spinBox_sdarkfield_m.value()
                for i, img in enumerate(imgs_ch1):
                    out, mask = self.model.spiral_darkfield(img, rho1, rho2, m)
                    self.results_ch1[i] = out
                    self.masks_ch1[i] = mask
        if imgs_ch2 is not None:
            n, h, w = imgs_ch2.shape
            self.results_ch2 = np.empty((n, h, w), dtype=np.float32)
            self.masks_ch2 = np.empty((n, h, w), dtype=np.uint8)
            if technique == 0:
                rho1 = self.view.spinBox_darkfield_rho1.value()
                rho2 = self.view.spinBox_darkfield_rho2.value()
                for i, img in enumerate(imgs_ch2):
                    out, mask = self.model.darfield(img, rho1, rho2)
                    self.results_ch2[i] = out
                    self.masks_ch2[i] = mask
            elif technique == 1:
                rho1 = self.view.spinBox_psdarkfield_rho1.value()
                rho2 = self.view.spinBox_psdarkfield_rho2.value()
                phaseshift = self.view.spinBox_psdarkfield_phaseshift.value()
                for i, img in enumerate(imgs_ch2):
                    out, mask = self.model.phase_shifted_darkfield(img, rho1, rho2, phaseshift)
                    self.results_ch2[i] = out
                    self.masks_ch2[i] = mask
            elif technique == 2:
                rho1 = self.view.spinBox_iwdarkfield_rho1.value()
                rho2 = self.view.spinBox_iwdarkfield_rho2.value()
                alpha = self.view.spinBox_iwdarkfield_alpha.value()
                for i, img in enumerate(imgs_ch2):
                    out, mask = self.model.intensity_weighted_darkfield(img, rho1, rho2, alpha)
                    self.results_ch2[i] = out
                    self.masks_ch2[i] = mask
            elif technique == 3:
                rho1 = self.view.spinBox_sdarkfield_rho1.value()
                rho2 = self.view.spinBox_sdarkfield_rho2.value()
                m = self.view.spinBox_sdarkfield_m.value()
                for i, img in enumerate(imgs_ch2):
                    out, mask = self.model.spiral_darkfield(img, rho1, rho2, m)
                    self.results_ch2[i] = out
                    self.masks_ch2[i] = mask
        # Always display both channels if available
        if self.results_ch1 is not None and self.results_ch1.shape[0] > 0:
            self.view.imageLabel_output1.axes.clear()
            self.view.imageLabel_output1.axes.imshow(self.results_ch1[0], cmap='gray')
            self.view.imageLabel_output1.draw()
            self.view.saveButton.setEnabled(True)
            if self.masks_ch1 is not None and self.masks_ch1.shape[0] > 0 and hasattr(self.view, 'imageLabel_preview'):
                self.view.imageLabel_preview.axes.clear()
                self.view.imageLabel_preview.axes.imshow(self.masks_ch1[0], cmap='gray')
                self.view.imageLabel_preview.draw()
            print(f"Displayed result and mask for channel 1, technique {self.TECHNIQUE_MODEL_MAP.get(technique)}.")
        if self.results_ch2 is not None and self.results_ch2.shape[0] > 0:
            self.view.imageLabel_output2.axes.clear()
            self.view.imageLabel_output2.axes.imshow(self.results_ch2[0], cmap='gray')
            self.view.imageLabel_output2.draw()
            self.view.saveButton.setEnabled(True)
            if self.masks_ch2 is not None and self.masks_ch2.shape[0] > 0 and hasattr(self.view, 'imageLabel_preview'):
                self.view.imageLabel_preview.axes.clear()
                self.view.imageLabel_preview.axes.imshow(self.masks_ch2[0], cmap='gray')
                self.view.imageLabel_preview.draw()
            print(f"Displayed result and mask for channel 2, technique {self.TECHNIQUE_MODEL_MAP.get(technique)}.")
        if (self.results_ch1 is None or self.results_ch1.shape[0] == 0) and (self.results_ch2 is None or self.results_ch2.shape[0] == 0):
            print("No results to display.")

    def on_technique_changed(self, index):
        self.technique_index = index
        if self.view is not None:
            self.view.stackedWidget_params.setCurrentIndex(index)
        print(f"Technique changed to {index}, model: {self.TECHNIQUE_MODEL_MAP.get(index)}")
        # Auto-recompute on change
        self.on_compute_clicked()

    def on_reference_changed(self, index):
        self.reference_index = index
        print(f"Reference changed to {index}")
        # Auto-recompute on change
        self.on_compute_clicked()

    def on_darkfield_rho1_changed(self, value):
        pass

    def on_darkfield_rho2_changed(self, value):
        pass

    def on_psdarkfield_rho1_changed(self, value):
        pass

    def on_psdarkfield_rho2_changed(self, value):
        pass

    def on_psdarkfield_phaseshift_changed(self, value):
        pass

    def on_iwdarkfield_rho1_changed(self, value):
        pass

    def on_iwdarkfield_rho2_changed(self, value):
        pass

    def on_sdarkfield_rho1_changed(self, value):
        pass

    def on_sdarkfield_rho2_changed(self, value):
        pass

    def on_sdarkfield_m_changed(self, value):
        pass

    def set_rho2_constraints(self):
        imgs = None
        if hasattr(self, 'imgs_ch1') and self.imgs_ch1 is not None and self.imgs_ch1.shape[0] > 0:
            imgs = self.imgs_ch1
        elif hasattr(self, 'imgs_ch2') and self.imgs_ch2 is not None and self.imgs_ch2.shape[0] > 0:
            imgs = self.imgs_ch2
        if imgs is not None:
            h, w = imgs[0].shape[:2]
            max_rho2 = int(min(h, w) // 2)
            for sb in [ self.view.spinBox_darkfield_rho2,
                        self.view.spinBox_psdarkfield_rho2,
                        self.view.spinBox_iwdarkfield_rho2,
                        self.view.spinBox_sdarkfield_rho2]:
                if sb.value() > max_rho2 or sb.value() < 1:
                    sb.setMaximum(max_rho2)
                    sb.setValue(max_rho2)
