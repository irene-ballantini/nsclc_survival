#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import SimpleITK as sitk
from pathlib import Path
from rt_utils import RTStructBuilder

class RadiomicsPreprocessor: 
    """
    Preprocessor for converting clinical data to NIfTI format (.nii.gz):
        - Uses rt_utils to convert contour points (vectors) from RTSTRUCT into a binary mask (numpy array with pixels).
        - Uses SimpleITK to convert the binary mask into a spatial volume (SimpleITK image with voxels) and saves it as .nii.gz.
        
    """

    def __init__(self, organized_path, preprocessed_path):
        self.organized_path = Path(organized_path)
        self.preprocessed_path = Path(preprocessed_path)
        self.preprocessed_path.mkdir(parents=True, exist_ok=True)

    def process_all_patients(self):
        patient_folders = sorted([f for f in self.organized_path.iterdir() if f.is_dir()])
        print(f"Found {len(patient_folders)} patients.")

        processed_successfully = []

        for i, patient_dir in enumerate(patient_folders, start=1):
            patient_id = patient_dir.name

            # Check if file already exists to avoid re-processing
            output_check = self.preprocessed_path / patient_id / "label.nii.gz"
            if output_check.exists():
                print(f"\n[{i}/{len(patient_folders)}] Skipping {patient_id}: Already processed.")
                continue

            print(f"\nProcessing folder {i}/{len(patient_folders)}: Patient {patient_id}")
            result = self.process_patient(patient_dir, patient_id)

            if result:
                print(f"--- SUCCESS: {patient_id} saved correctly.")
                processed_successfully.append(result)
            else:
                print(f"--- FAILED: {patient_id} was skipped.")

        if len(processed_successfully) != 0:
            print(f"\nPre-processing completed! {len(processed_successfully)}/{len(patient_folders)} patients processed.")
        else:
            print("\nPatients were already processed or skipped due to errors. No new files created.")

        print(f"The NIfTI files are in: {self.preprocessed_path}")
    
    def process_patient(self, patient_dir, patient_id, roi_target="GTV-1"): 

        """
        Process a single patient directory to extract the CT and the tumor mask, and save them in NIfTI format.

        GTV-1 = label for Primary Gross Tumor Volume

        Args:
            patient_dir (Path): path to the patient's folder containing the CT and RTSTRUCT subfolders.
            patient_id (str): ID of the patient (used for naming the output folder).
            roi_target (str): Name of the ROI to extract. Defaults to "GTV-1".

        Returns:
            str | None: patient_id if processed successfully, None if failed
        """
      
        path_ct = patient_dir / "CT"
        #Note: RTSTRUCT file usually has a single file
        rt_files = list((patient_dir / "RTSTRUCT").glob("*.dcm")) 

        if not rt_files:
            print(f"Skipping {patient_id}: No file RTSTRUCT found.")
            return None
        
        # Take the first RTSTRUCT file found
        path_rtstruct = rt_files[0]  

        print(f"CT Path: {path_ct}")
        print(f"RT Path: {path_rtstruct}")

        try:
            # Load the RTSTRUCT 
            rtstruct = RTStructBuilder.create_from(
                dicom_series_path=str(path_ct), 
                rt_struct_path=str(path_rtstruct),
            )

            rois = rtstruct.get_roi_names()
            print(f"ROI found for this patient {patient_id}: {rois}")

            if not rois:
                print("WARNING: No ROI loaded correctly.")
                return None

            if roi_target not in rois:
                print(f"Skipping {patient_id}: ROI '{roi_target}' not found. Available ROIs: {rois}")
                return None
        
            # Extract the tumor mask 
            tumor_mask = rtstruct.get_roi_mask_by_name(roi_target)  

            # Verify the dimensions of the mask
            print(f"Shape of the mask: {tumor_mask.shape}") 
            # Should be (height, width, number_of_slices)
        
            # Load the CT series as a SimpleITK image
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(path_ct))
            reader.SetFileNames(dicom_names)
            ct_image = reader.Execute()

            # Now the object ct_image "knows" the size of the voxels (e.g., 1mm x 1mm x 3mm)
            
            # 1. Transform the tumor mask from numpy array to SimpleITK image
            tumor_mask_itk = self._numpy_to_itk(tumor_mask, ct_image)

            # 2. Resample CT to isotropic spacing (e.g., 1mm x 1mm x 1mm)
            print(f"Resampling {patient_id} to 1.0mm isotropic...")
            ct_resampled = self._resample_image(ct_image, is_label=False)

            # Resample the tumor mask using the ct_resampled
            mask_resampled = self._resample_mask_to_reference(tumor_mask_itk, ct_resampled)       

            # 3. Saving in NIfTI format
            output_patient_dir = self.preprocessed_path / patient_id
            output_patient_dir.mkdir(parents=True, exist_ok=True)

            sitk.WriteImage(ct_resampled, str(output_patient_dir / "image.nii.gz"))
            sitk.WriteImage(mask_resampled, str(output_patient_dir / "label.nii.gz"))

            return patient_id

        except Exception as e:
            print(f"Skipping patient {patient_id} due to RTSTRUCT error: {e}")
            return None
    
    def _numpy_to_itk(self, mask_np, reference_image):
        """ 
        Convert a numpy array mask (H, W, Slices) to a SimpleITK image (Slices, H, W), ensuring correct spatial metadata.

        Args:
            mask_np (np.ndarray): Boolean numpy array provided by rt_utils, with shape (H, W, Slices).
            reference_image (sitk.Image): Original CT image to copy the spatial metadata

        Returns:
            sitk.Image: Mask as SimpleITK image with correct spatial metadata.
        """
        
        mask_itk = sitk.GetImageFromArray(mask_np.astype(np.uint8).transpose(2, 0, 1))
    
        # Copy spatial metadata from the original CT
        mask_itk.SetSpacing(reference_image.GetSpacing())
        mask_itk.SetOrigin(reference_image.GetOrigin())
        mask_itk.SetDirection(reference_image.GetDirection())
        return mask_itk
    
    def _resample_image(self, itk_image, out_spacing = [1.0, 1.0, 1.0], is_label= False):
        """
        Resample a SimpleITK image to a new spacing.

        Args:
            itk_image (sitk.Image): Image to resample.
            out_spacing (list): New spacing in mm. Defaults to [1.0, 1.0, 1.0].
            is_label (bool):If True, use NearestNeighbor to preserve binary values.
                            If False, use BSpline to maintain image quality. 
                            Defaults to False.

        Returns:
            sitk.Image: Resampled image.
        """

        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()
        
        # Calculate the new size in pixels (Size) to maintain the same physical volume
        out_size = [
            int(round(original_size[i] * (original_spacing[i] / out_spacing[i])))
            for i in range(3)
        ]
        
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        
        # Choose the interpolation method
        if is_label:
            # For the binary mask: we don't want average values (e.g., 0.5), only 0 or 1
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            # For the CT: BSpline ensures better image quality after resampling
            resample.SetInterpolator(sitk.sitkBSpline)
            
        return resample.Execute(itk_image)
    
    def _resample_mask_to_reference(self, mask_itk, reference_itk):
        """
        Resample the mask to match the spatial properties of the reference image (CT)

        Args:
            mask_itk (sitk.Image): Mask (label) to resample.
            reference_itk (sitk.Image): Resampled image (CT) to use as reference.

        Returns:
            sitk.Image: Resampled mask.
        """
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(reference_itk) 
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetTransform(sitk.Transform())
    
        return resample.Execute(mask_itk)