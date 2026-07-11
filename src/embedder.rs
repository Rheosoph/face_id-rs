use crate::error::FaceIdError;
#[cfg(feature = "hf-hub")]
use crate::model_manager::{HfModel, get_hf_model};
use bon::bon;
use image::{ImageBuffer, Rgb};
use ndarray::{Array2, Array4, s};
use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

pub struct ArcFaceEmbedder {
    pub session: Session,
    pub input_name: String,
}

#[bon]
impl ArcFaceEmbedder {
    #[cfg(feature = "hf-hub")]
    #[builder(finish_fn = build)]
    pub async fn from_hf(
        #[builder(default = HfModel::default_embedder())] model: HfModel,
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
            .ok_or_else(|| FaceIdError::InvalidModel("Embedder has no inputs".into()))?
            .name()
            .to_string();
        if session.outputs().is_empty() {
            return Err(FaceIdError::InvalidModel("Embedder has no outputs".into()));
        }

        Ok(Self {
            session,
            input_name,
        })
    }

    pub fn compute_embeddings_batch(
        &mut self,
        aligned_imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Vec<Vec<f32>>, FaceIdError> {
        if aligned_imgs.is_empty() {
            return Ok(vec![]);
        }

        let input_tensor = Self::create_input_tensor_batch(aligned_imgs)?;
        let input_value = Value::from_array(input_tensor)?;

        let outputs = self
            .session
            .run(ort::inputs![&self.input_name => input_value])?;

        let output = outputs
            .values()
            .next()
            .ok_or_else(|| FaceIdError::InvalidModel("Embedder produced no outputs".into()))?;
        let mut output_tensor = output
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()?;

        let expected_batch_size = aligned_imgs.len();
        if output_tensor.shape()[0] != expected_batch_size {
            return Err(FaceIdError::Ort(format!(
                "Embedder batch size mismatch: expected {expected_batch_size}, got {}",
                output_tensor.shape()[0]
            )));
        }
        if output_tensor.shape()[1] == 0 {
            return Err(FaceIdError::InvalidModel(
                "Embedder produced empty embeddings".into(),
            ));
        }
        if output_tensor.iter().any(|value| !value.is_finite()) {
            return Err(FaceIdError::InvalidModel(
                "Embedder produced non-finite values".into(),
            ));
        }

        Self::l2_normalize_batch(&mut output_tensor);

        let batch_size = output_tensor.shape()[0];
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            results.push(output_tensor.slice(s![i, ..]).to_vec());
        }

        Ok(results)
    }

    fn create_input_tensor_batch(
        imgs: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    ) -> Result<Array4<f32>, FaceIdError> {
        let batch_size = imgs.len();
        // Shape: [N, 3, 112, 112]
        let mut array = Array4::<f32>::zeros((batch_size, 3, 112, 112));

        let data = array
            .as_slice_memory_order_mut()
            .ok_or_else(|| FaceIdError::Ort("Failed to get mutable slice".into()))?;

        let channel_stride = 112 * 112;
        for (batch_idx, img) in imgs.iter().enumerate() {
            let (w, h) = img.dimensions();
            if w != 112 || h != 112 {
                return Err(FaceIdError::InvalidModel(format!(
                    "ArcFace requires 112x112 input, got {w}x{h}"
                )));
            }

            let raw = img.as_raw();
            let batch_offset = batch_idx * 3 * channel_stride;

            for (i, chunk) in raw.chunks_exact(3).enumerate() {
                data[batch_offset + i] = (f32::from(chunk[0]) - 127.5) / 127.5;
                data[batch_offset + i + channel_stride] = (f32::from(chunk[1]) - 127.5) / 127.5;
                data[batch_offset + i + 2 * channel_stride] = (f32::from(chunk[2]) - 127.5) / 127.5;
            }
        }

        Ok(array)
    }

    /// Takes an ALIGNED face image (112x112) and returns a normalized 512-d embedding.
    pub fn compute_embedding(
        &mut self,
        aligned_img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Vec<f32>, FaceIdError> {
        let mut results = self.compute_embeddings_batch(std::slice::from_ref(aligned_img))?;
        results
            .pop()
            .ok_or_else(|| FaceIdError::Ort("Embedder failed to produce an embedding".into()))
    }

    /// Preprocessing wrapper for a single image.
    pub fn create_input_tensor(
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<Array4<f32>, FaceIdError> {
        Self::create_input_tensor_batch(std::slice::from_ref(img))
    }

    /// Performs L2 normalization on every row of an `[N, Dim]` embedding array in-place.
    pub fn l2_normalize_batch(embeddings: &mut Array2<f32>) {
        for mut embedding in embeddings.rows_mut() {
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                embedding *= 1.0 / norm;
            }
        }
    }

    /// Normalizes a single vector.
    pub fn l2_normalize(vec: &mut [f32]) {
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            let inv_norm = 1.0 / norm;
            for x in vec.iter_mut() {
                *x *= inv_norm;
            }
        }
    }

    /// Computes cosine similarity between two L2-normalized embeddings.
    /// Range: -1.0 to 1.0 (Higher is more similar).
    #[must_use]
    pub fn compute_similarity(emb1: &[f32], emb2: &[f32]) -> f32 {
        Self::try_compute_similarity(emb1, emb2).unwrap_or(f32::NAN)
    }

    /// Computes cosine similarity after validating that both embeddings have the same size.
    ///
    /// Prefer this checked variant when embeddings may come from different models. The legacy
    /// [`Self::compute_similarity`] method returns `NaN` for invalid input instead of silently
    /// truncating the longer vector.
    pub fn try_compute_similarity(emb1: &[f32], emb2: &[f32]) -> Result<f32, FaceIdError> {
        if emb1.is_empty() || emb1.len() != emb2.len() {
            return Err(FaceIdError::InvalidModel(format!(
                "Embedding lengths must be equal and non-zero, got {} and {}",
                emb1.len(),
                emb2.len()
            )));
        }
        if emb1.iter().chain(emb2).any(|value| !value.is_finite()) {
            return Err(FaceIdError::InvalidModel(
                "Embeddings must contain only finite values".into(),
            ));
        }
        Ok(emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_similarity_rejects_different_lengths() {
        assert!(ArcFaceEmbedder::try_compute_similarity(&[1.0], &[1.0, 0.0]).is_err());
        assert!(ArcFaceEmbedder::compute_similarity(&[1.0], &[1.0, 0.0]).is_nan());
    }

    #[test]
    fn batch_normalization_handles_zero_rows() {
        let mut embeddings = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        ArcFaceEmbedder::l2_normalize_batch(&mut embeddings);
        assert!((embeddings[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((embeddings[[0, 1]] - 0.8).abs() < 1e-6);
        assert_eq!(embeddings.row(1).as_slice().unwrap(), &[0.0, 0.0]);
    }
}
