#![allow(clippy::similar_names)]
#[cfg(feature = "clustering")]
use crate::analyzer::{FaceAnalysis, FaceAnalyzer};
use crate::detector::BoundingBox;
#[cfg(feature = "clustering")]
use crate::error::FaceIdError;
#[cfg(feature = "clustering")]
use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams, NnAlgorithm};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
#[cfg(feature = "clustering")]
use rayon::prelude::*;
#[cfg(feature = "clustering")]
use std::collections::HashMap;
#[cfg(feature = "clustering")]
use std::path::{Path, PathBuf};

/// Extracts a square, padded thumbnail for a face.
///
/// # Arguments
/// * `img` - The source image.
/// * `bbox` - The detected face bounding box.
/// * `padding_factor` - How much context to show. 1.0 = tight crop, 2.0 = face takes up 50% of width. (1.5 - 1.8 is usually ideal for UI).
/// * `size` - The output resolution (e.g., 256 for a 256x256 thumbnail).
#[must_use]
pub fn extract_face_thumbnail(
    img: &DynamicImage,
    bbox: &BoundingBox,
    padding_factor: f32,
    size: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (img_w, img_h) = img.dimensions();
    let mut output = ImageBuffer::new(size, size);
    if size == 0
        || img_w == 0
        || img_h == 0
        || !padding_factor.is_finite()
        || padding_factor <= 0.0
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
    let side = width.max(height) * padding_factor;
    let crop_x = cx - side * 0.5;
    let crop_y = cy - side * 0.5;
    let sample_scale = side / size as f32;

    for py in 0..size {
        let src_y = (py as f32 + 0.5).mul_add(sample_scale, crop_y) - 0.5;
        for px in 0..size {
            let src_x = (px as f32 + 0.5).mul_add(sample_scale, crop_x) - 0.5;
            output.put_pixel(
                px,
                py,
                bilinear_sample_dynamic(img, src_x, src_y, img_w, img_h),
            );
        }
    }

    output
}

#[inline]
fn bilinear_sample_dynamic(img: &DynamicImage, x: f32, y: f32, width: u32, height: u32) -> Rgb<u8> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= width as f32
        || y >= height as f32
    {
        return Rgb([0, 0, 0]);
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p00 = img.get_pixel(x0, y0);
    let p10 = img.get_pixel(x1, y0);
    let p01 = img.get_pixel(x0, y1);
    let p11 = img.get_pixel(x1, y1);

    Rgb([
        bilerp(p00[0], p10[0], p01[0], p11[0], fx, fy),
        bilerp(p00[1], p10[1], p01[1], p11[1], fx, fy),
        bilerp(p00[2], p10[2], p01[2], p11[2], fx, fy),
    ])
}

#[inline]
fn bilerp(c00: u8, c10: u8, c01: u8, c11: u8, fx: f32, fy: f32) -> u8 {
    let top = (f32::from(c10) - f32::from(c00)).mul_add(fx, f32::from(c00));
    let bottom = (f32::from(c11) - f32::from(c01)).mul_add(fx, f32::from(c01));
    (bottom - top).mul_add(fy, top) as u8
}

/// Clusters faces from a list of images using the HDBSCAN algorithm.
///
/// This function performs the following steps:
/// 1. Loads each image from the provided paths.
/// 2. Performs face analysis (detection and embedding) using the provided `FaceAnalyzer`.
/// 3. Clusters the resulting face embeddings using HDBSCAN.
///
/// Returns a mapping of cluster IDs to a list of (image path, face analysis) pairs.
/// Cluster ID -1 represents noise.
///
/// # Errors
/// Returns a `FaceIdError` if any image fails to load, analysis fails, or clustering fails.
///
/// # Feature Gated
/// This function is only available when the `clustering` feature is enabled.
#[cfg(feature = "clustering")]
#[bon::builder]
pub fn cluster_faces<P: AsRef<Path> + Sync + Send>(
    #[builder(start_fn)] analyzer: &FaceAnalyzer,
    #[builder(start_fn)] paths: Vec<P>,
    #[builder(default = 5)] min_cluster_size: usize,
    #[builder(default = usize::MAX)] max_cluster_size: usize,
    #[builder(default = false)] allow_single_cluster: bool,
    min_samples: Option<usize>,
    #[builder(default = 0.0)] epsilon: f64,
    #[builder(default = DistanceMetric::Euclidean)] dist_metric: DistanceMetric,
    #[builder(default = NnAlgorithm::Auto)] nn_algo: NnAlgorithm,
) -> Result<HashMap<i32, Vec<(PathBuf, FaceAnalysis)>>, FaceIdError> {
    let mut all_faces: Vec<(PathBuf, FaceAnalysis)> = paths
        .into_par_iter()
        .map(
            |path_ref| -> Result<Vec<(PathBuf, FaceAnalysis)>, FaceIdError> {
                let path = path_ref.as_ref().to_path_buf();
                let img = image::open(&path)?;
                let faces = analyzer.analyze(&img)?;
                Ok(faces.into_iter().map(|f| (path.clone(), f)).collect())
            },
        )
        .collect::<Result<Vec<Vec<_>>, _>>()?
        .into_iter()
        .flatten()
        .collect();

    if all_faces.is_empty() {
        return Ok(HashMap::new());
    }

    // Move embeddings into HDBSCAN's contiguous input without cloning the 512-value vectors.
    // They are moved back into their corresponding analyses after clustering.
    let embeddings: Vec<Vec<f32>> = all_faces
        .iter_mut()
        .map(|(_, face)| std::mem::take(&mut face.embedding))
        .collect();

    if embeddings.is_empty() {
        return Ok(HashMap::new());
    }

    let mut hp_builder = HdbscanHyperParams::builder()
        .min_cluster_size(min_cluster_size)
        .max_cluster_size(max_cluster_size)
        .allow_single_cluster(allow_single_cluster)
        .epsilon(epsilon)
        .dist_metric(dist_metric)
        .nn_algorithm(nn_algo);

    if let Some(ms) = min_samples {
        hp_builder = hp_builder.min_samples(ms);
    } else {
        hp_builder = hp_builder.min_samples(min_cluster_size);
    }

    let hyper_params = hp_builder.build();
    let clusterer = Hdbscan::new(&embeddings, hyper_params);
    let labels: Vec<i32> = clusterer
        .cluster()
        .map_err(|e| FaceIdError::Clustering(e.to_string()))?;
    if labels.len() != all_faces.len() {
        return Err(FaceIdError::Clustering(format!(
            "Expected {} labels, got {}",
            all_faces.len(),
            labels.len()
        )));
    }

    let mut clusters: HashMap<i32, Vec<(PathBuf, FaceAnalysis)>> = HashMap::new();
    for (((path, mut face), embedding), label) in all_faces.into_iter().zip(embeddings).zip(labels)
    {
        face.embedding = embedding;
        clusters.entry(label).or_default().push((path, face));
    }

    Ok(clusters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_extract_face_thumbnail_edge_case() {
        // 50x100 white image
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(50, 100, Rgb([255, 255, 255])));

        // Face near the right edge, so with padding/aspect ratio 1 it would stick out
        let bbox = BoundingBox {
            x1: 0.8,
            y1: 0.5,
            x2: 1.0,
            y2: 0.6,
        };

        let thumbnail = extract_face_thumbnail(&img, &bbox, 4.0, 100);

        assert_eq!(thumbnail.width(), 100);
        assert_eq!(thumbnail.height(), 100);
    }
}
