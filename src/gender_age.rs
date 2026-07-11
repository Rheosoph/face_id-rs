#![allow(clippy::similar_names)]
use crate::detector::BoundingBox;
use crate::error::FaceIdError;
use crate::face_align::bilinear_sample;
#[cfg(feature = "hf-hub")]
use crate::model_manager::{HfModel, get_hf_model};
use bon::bon;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array4;
use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Gender {
    Female = 0,
    Male = 1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenderAge {
    pub gender: Gender,
    pub age: u8,
}

pub struct GenderAgeEstimator {
    pub session: Session,
    pub input_name: String,
}

#[bon]
impl GenderAgeEstimator {
    #[cfg(feature = "hf-hub")]
    #[builder(finish_fn = build)]
    pub async fn from_hf(
        #[builder(default = HfModel::default_gender_age())] model: HfModel,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceIdError> {
        let model_path = get_hf_model(model).await?;
        Self::builder(model_path)
            .with_execution_providers(with_execution_providers)
            .build()
    }

    #[builder]
    pub fn new(
        #[builder(start_fn)] model_path: impl AsRef<Path>,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, FaceIdError> {
        let session = Session::builder()?
            .with_execution_providers(with_execution_providers)?
            .commit_from_file(model_path)?;

        let input_name = session
            .inputs()
            .first()
            .ok_or_else(|| FaceIdError::InvalidModel("Gender/age model has no inputs".into()))?
            .name()
            .to_string();
        if session.outputs().is_empty() {
            return Err(FaceIdError::InvalidModel(
                "Gender/age model has no outputs".into(),
            ));
        }

        Ok(Self {
            session,
            input_name,
        })
    }

    /// Estimates gender and age for a batch of cropped face images.
    pub fn estimate_batch(
        &mut self,
        face_imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Vec<GenderAge>, FaceIdError> {
        if face_imgs.is_empty() {
            return Ok(vec![]);
        }

        let input_tensor = Self::create_input_tensor_batch(face_imgs)?;
        let input_value = Value::from_array(input_tensor)?;
        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => input_value])?;

        let output = outputs.values().next().ok_or_else(|| {
            FaceIdError::InvalidModel("Gender/age model produced no outputs".into())
        })?;
        let output_tensor = output.try_extract_array::<f32>()?;
        let batch_size = face_imgs.len();

        if output_tensor.ndim() != 2
            || output_tensor.shape()[0] != batch_size
            || output_tensor.shape()[1] != 3
        {
            return Err(FaceIdError::Ort(format!(
                "GenderAge output shape mismatch: expected [{batch_size}, 3], got {:?}",
                output_tensor.shape()
            )));
        }

        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let prob_female = output_tensor[[i, 0]];
            let prob_male = output_tensor[[i, 1]];
            let age_raw = output_tensor[[i, 2]];

            if !prob_female.is_finite() || !prob_male.is_finite() || !age_raw.is_finite() {
                return Err(FaceIdError::InvalidModel(format!(
                    "Gender/age model produced non-finite values for batch item {i}"
                )));
            }

            let gender = if prob_male > prob_female {
                Gender::Male
            } else {
                Gender::Female
            };
            let age = (age_raw * 100.0).round().clamp(0.0, 100.0) as u8;
            results.push(GenderAge { gender, age });
        }

        Ok(results)
    }

    /// Estimates gender and age from an image and a detected face bounding box.
    pub fn estimate(
        &mut self,
        img: &DynamicImage,
        bbox: &BoundingBox,
    ) -> Result<GenderAge, FaceIdError> {
        let rgb_img = img.to_rgb8();
        let cropped_face = Self::align_crop(&rgb_img, bbox, 96);
        let results = self.estimate_batch(&[cropped_face])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| FaceIdError::Ort("GenderAge failed to produce output".into()))
    }

    /// `InsightFace` Attribute alignment: Creates a square crop based on the `BBox`
    /// with a 1.5x expansion factor to include context.
    #[must_use]
    pub fn align_crop(
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        bbox: &BoundingBox,
        output_size: u32,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (img_w, img_h) = img.dimensions();
        let mut output = ImageBuffer::new(output_size, output_size);
        if output_size == 0
            || img_w == 0
            || img_h == 0
            || !bbox.x1.is_finite()
            || !bbox.y1.is_finite()
            || !bbox.x2.is_finite()
            || !bbox.y2.is_finite()
        {
            return output;
        }

        let x1 = bbox.x1.clamp(0.0, 1.0) * img_w as f32;
        let y1 = bbox.y1.clamp(0.0, 1.0) * img_h as f32;
        let x2 = bbox.x2.clamp(0.0, 1.0) * img_w as f32;
        let y2 = bbox.y2.clamp(0.0, 1.0) * img_h as f32;
        let width = x2 - x1;
        let height = y2 - y1;
        if width <= 0.0 || height <= 0.0 {
            return output;
        }

        let cx = (x1 + x2) * 0.5;
        let cy = (y1 + y2) * 0.5;
        let side = width.max(height) * 1.5;
        let sample_scale = side / output_size as f32;
        let output_center = output_size as f32 * 0.5;

        // Sample directly into the model-sized output using the same integer-coordinate mapping
        // as OpenCV's `warpAffine`. This keeps memory bounded and pads out-of-bounds pixels black.
        for py in 0..output_size {
            let src_y = (py as f32 - output_center).mul_add(sample_scale, cy);
            for px in 0..output_size {
                let src_x = (px as f32 - output_center).mul_add(sample_scale, cx);
                output.put_pixel(px, py, bilinear_sample(img, src_x, src_y, img_w, img_h));
            }
        }

        output
    }

    fn create_input_tensor_batch(
        imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Array4<f32>, FaceIdError> {
        let batch_size = imgs.len();
        let mut array = Array4::<f32>::zeros((batch_size, 3, 96, 96));

        let data = array.as_slice_memory_order_mut().ok_or_else(|| {
            FaceIdError::FailedToGetMutableSlice("Failed to get mutable slice from array".into())
        })?;

        let channel_stride = 96 * 96;
        for (batch_idx, img) in imgs.iter().enumerate() {
            let (w, h) = img.dimensions();
            if w != 96 || h != 96 {
                return Err(FaceIdError::InvalidModel(format!(
                    "GenderAge requires 96x96 input, got {w}x{h}"
                )));
            }

            let raw = img.as_raw();
            let batch_offset = batch_idx * 3 * channel_stride;

            // InsightFace creates an RGB blob (`swapRB=true`) from OpenCV's BGR image.
            // `image` already stores RGB, so preserve the channel order here.
            for (i, chunk) in raw.chunks_exact(3).enumerate() {
                data[batch_offset + i] = f32::from(chunk[0]); // R
                data[batch_offset + i + channel_stride] = f32::from(chunk[1]); // G
                data[batch_offset + i + 2 * channel_stride] = f32::from(chunk[2]); // B
            }
        }

        Ok(array)
    }

    pub fn create_input_tensor(
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Array4<f32>, FaceIdError> {
        Self::create_input_tensor_batch(std::slice::from_ref(img))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_tensor_preserves_rgb_channel_order() {
        let mut img = ImageBuffer::from_pixel(96, 96, Rgb([0, 0, 0]));
        img.put_pixel(0, 0, Rgb([10, 20, 30]));

        let tensor = GenderAgeEstimator::create_input_tensor(&img).unwrap();

        assert!((tensor[[0, 0, 0, 0]] - 10.0).abs() < f32::EPSILON);
        assert!((tensor[[0, 1, 0, 0]] - 20.0).abs() < f32::EPSILON);
        assert!((tensor[[0, 2, 0, 0]] - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn crop_memory_is_bounded_by_output_size() {
        let img = ImageBuffer::from_pixel(2, 2, Rgb([255, 255, 255]));
        let bbox = BoundingBox {
            x1: -1.0e20,
            y1: -1.0e20,
            x2: 1.0e20,
            y2: 1.0e20,
        };

        let crop = GenderAgeEstimator::align_crop(&img, &bbox, 96);

        assert_eq!(crop.dimensions(), (96, 96));
    }
}
