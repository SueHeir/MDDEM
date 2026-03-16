//! Spatial region primitives for groups, particle insertion, and wall definitions.

use rand::Rng;
use serde::Deserialize;

/// Axis for cylinder orientation.
#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    X,
    Y,
    Z,
}

/// A spatial region primitive. Deserialized from TOML with `type` tag.
///
/// # Examples
/// ```toml
/// region = { type = "block", min = [0, 0, 0], max = [1, 1, 1] }
/// region = { type = "sphere", center = [0, 0, 0], radius = 5.0 }
/// region = { type = "cylinder", center = [0, 0], radius = 1.0, axis = "z", lo = 0.0, hi = 5.0 }
/// region = { type = "plane", point = [0, 0, 0], normal = [0, 0, 1] }
/// ```
#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Region {
    Block {
        min: [f64; 3],
        max: [f64; 3],
    },
    Sphere {
        center: [f64; 3],
        radius: f64,
    },
    Cylinder {
        center: [f64; 2],
        radius: f64,
        axis: Axis,
        lo: f64,
        hi: f64,
    },
    Plane {
        point: [f64; 3],
        normal: [f64; 3],
    },
    Union {
        regions: Vec<Region>,
    },
    Intersect {
        regions: Vec<Region>,
    },
}

impl Region {
    /// Test whether a point lies inside (or on) the region.
    ///
    /// For `Plane`, returns true if the point is on the positive side (dot product >= 0).
    pub fn contains(&self, pos: &[f64; 3]) -> bool {
        match self {
            Region::Block { min, max } => {
                pos[0] >= min[0]
                    && pos[0] <= max[0]
                    && pos[1] >= min[1]
                    && pos[1] <= max[1]
                    && pos[2] >= min[2]
                    && pos[2] <= max[2]
            }
            Region::Sphere { center, radius } => {
                let dx = pos[0] - center[0];
                let dy = pos[1] - center[1];
                let dz = pos[2] - center[2];
                dx * dx + dy * dy + dz * dz <= radius * radius
            }
            Region::Cylinder {
                center,
                radius,
                axis,
                lo,
                hi,
            } => {
                let (axial, r0, r1) = match axis {
                    Axis::X => (pos[0], pos[1] - center[0], pos[2] - center[1]),
                    Axis::Y => (pos[1], pos[0] - center[0], pos[2] - center[1]),
                    Axis::Z => (pos[2], pos[0] - center[0], pos[1] - center[1]),
                };
                axial >= *lo && axial <= *hi && r0 * r0 + r1 * r1 <= radius * radius
            }
            Region::Plane {
                point,
                normal,
            } => {
                let dx = pos[0] - point[0];
                let dy = pos[1] - point[1];
                let dz = pos[2] - point[2];
                dx * normal[0] + dy * normal[1] + dz * normal[2] >= 0.0
            }
            Region::Union { regions } => regions.iter().any(|r| r.contains(pos)),
            Region::Intersect { regions } => regions.iter().all(|r| r.contains(pos)),
        }
    }

    /// Generate a uniformly random point inside the region.
    ///
    /// # Panics
    /// Panics for `Plane` (infinite, no bounded volume).
    pub fn random_point_inside(&self, rng: &mut impl Rng) -> [f64; 3] {
        match self {
            Region::Block { min, max } => [
                rng.random_range(min[0]..max[0]),
                rng.random_range(min[1]..max[1]),
                rng.random_range(min[2]..max[2]),
            ],
            Region::Sphere { center, radius } => {
                // Rejection sampling for uniform distribution in sphere
                loop {
                    let x = rng.random_range(-1.0..1.0);
                    let y = rng.random_range(-1.0..1.0);
                    let z = rng.random_range(-1.0..1.0);
                    if x * x + y * y + z * z <= 1.0 {
                        return [
                            center[0] + x * radius,
                            center[1] + y * radius,
                            center[2] + z * radius,
                        ];
                    }
                }
            }
            Region::Cylinder {
                center,
                radius,
                axis,
                lo,
                hi,
            } => {
                // Rejection sampling for uniform distribution in cylinder cross-section
                loop {
                    let u = rng.random_range(-1.0..1.0);
                    let v = rng.random_range(-1.0..1.0);
                    if u * u + v * v <= 1.0 {
                        let axial = rng.random_range(*lo..*hi);
                        let r0 = center[0] + u * radius;
                        let r1 = center[1] + v * radius;
                        return match axis {
                            Axis::X => [axial, r0, r1],
                            Axis::Y => [r0, axial, r1],
                            Axis::Z => [r0, r1, axial],
                        };
                    }
                }
            }
            Region::Plane { .. } => {
                panic!("Cannot generate random point inside a Plane region (unbounded)");
            }
            Region::Union { regions } => {
                if regions.is_empty() {
                    panic!("Cannot generate random point inside empty Union");
                }
                // Pick a random child region and sample from it
                loop {
                    let idx = rng.random_range(0..regions.len());
                    let pt = regions[idx].random_point_inside(rng);
                    // Point from any child is valid for union
                    return pt;
                }
            }
            Region::Intersect { regions } => {
                if regions.is_empty() {
                    panic!("Cannot generate random point inside empty Intersect");
                }
                // Rejection sampling: sample from first child, accept if in all
                loop {
                    let pt = regions[0].random_point_inside(rng);
                    if regions.iter().all(|r| r.contains(&pt)) {
                        return pt;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_contains() {
        let r = Region::Block {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        assert!(r.contains(&[0.5, 0.5, 0.5]));
        assert!(r.contains(&[0.0, 0.0, 0.0]));
        assert!(r.contains(&[1.0, 1.0, 1.0]));
        assert!(!r.contains(&[1.1, 0.5, 0.5]));
        assert!(!r.contains(&[-0.1, 0.5, 0.5]));
    }

    #[test]
    fn test_sphere_contains() {
        let r = Region::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        assert!(r.contains(&[0.0, 0.0, 0.0]));
        assert!(r.contains(&[0.5, 0.5, 0.5]));
        assert!(!r.contains(&[1.0, 1.0, 0.0]));
    }

    #[test]
    fn test_cylinder_contains() {
        let r = Region::Cylinder {
            center: [0.0, 0.0],
            radius: 1.0,
            axis: Axis::Z,
            lo: 0.0,
            hi: 5.0,
        };
        assert!(r.contains(&[0.0, 0.0, 2.5]));
        assert!(r.contains(&[0.5, 0.5, 1.0]));
        assert!(!r.contains(&[0.0, 0.0, -1.0]));
        assert!(!r.contains(&[2.0, 0.0, 2.5]));
    }

    #[test]
    fn test_plane_contains() {
        let r = Region::Plane {
            point: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        };
        assert!(r.contains(&[0.0, 0.0, 1.0]));
        assert!(r.contains(&[0.0, 0.0, 0.0]));
        assert!(!r.contains(&[0.0, 0.0, -1.0]));
    }

    #[test]
    fn test_block_random_point() {
        let r = Region::Block {
            min: [1.0, 2.0, 3.0],
            max: [4.0, 5.0, 6.0],
        };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let p = r.random_point_inside(&mut rng);
            assert!(r.contains(&p), "random point {:?} not in block", p);
        }
    }

    #[test]
    fn test_sphere_random_point() {
        let r = Region::Sphere {
            center: [1.0, 2.0, 3.0],
            radius: 2.0,
        };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let p = r.random_point_inside(&mut rng);
            assert!(r.contains(&p), "random point {:?} not in sphere", p);
        }
    }

    #[test]
    #[should_panic(expected = "Cannot generate random point inside a Plane")]
    fn test_plane_random_panics() {
        let r = Region::Plane {
            point: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        };
        let mut rng = rand::rng();
        r.random_point_inside(&mut rng);
    }

    #[test]
    fn test_cylinder_contains_axis_x() {
        // Axis::X: axial = x, center is in (Y, Z) plane
        let r = Region::Cylinder {
            center: [2.0, 3.0], // center_y=2, center_z=3
            radius: 1.0,
            axis: Axis::X,
            lo: 0.0,
            hi: 5.0,
        };
        // On axis center: x=2.5, y=2.0, z=3.0
        assert!(r.contains(&[2.5, 2.0, 3.0]));
        // Near edge in Y: y=2.9, z=3.0 → r=0.9
        assert!(r.contains(&[1.0, 2.9, 3.0]));
        // Outside radius in Y: y=5.0, z=3.0 → r=3.0
        assert!(!r.contains(&[2.5, 5.0, 3.0]));
        // Outside axial range: x=-1.0
        assert!(!r.contains(&[-1.0, 2.0, 3.0]));
        // Outside axial range: x=6.0
        assert!(!r.contains(&[6.0, 2.0, 3.0]));
    }

    #[test]
    fn test_cylinder_contains_axis_y() {
        // Axis::Y: axial = y, center is in (X, Z) plane
        let r = Region::Cylinder {
            center: [1.0, 4.0], // center_x=1, center_z=4
            radius: 0.5,
            axis: Axis::Y,
            lo: -1.0,
            hi: 3.0,
        };
        // On axis center: x=1.0, y=0.0, z=4.0
        assert!(r.contains(&[1.0, 0.0, 4.0]));
        // Near edge: x=1.4, z=4.0 → r=0.4
        assert!(r.contains(&[1.4, 2.0, 4.0]));
        // Outside radius: x=2.0, z=4.0 → r=1.0
        assert!(!r.contains(&[2.0, 1.0, 4.0]));
        // Outside axial range: y=-2.0
        assert!(!r.contains(&[1.0, -2.0, 4.0]));
    }

    #[test]
    fn test_cylinder_random_point_axis_x() {
        let r = Region::Cylinder {
            center: [2.0, 3.0],
            radius: 1.5,
            axis: Axis::X,
            lo: -1.0,
            hi: 4.0,
        };
        let mut rng = rand::rng();
        for _ in 0..200 {
            let p = r.random_point_inside(&mut rng);
            assert!(r.contains(&p), "random point {:?} not in X-cylinder", p);
        }
    }

    #[test]
    fn test_cylinder_random_point_axis_y() {
        let r = Region::Cylinder {
            center: [1.0, 4.0],
            radius: 2.0,
            axis: Axis::Y,
            lo: 0.0,
            hi: 10.0,
        };
        let mut rng = rand::rng();
        for _ in 0..200 {
            let p = r.random_point_inside(&mut rng);
            assert!(r.contains(&p), "random point {:?} not in Y-cylinder", p);
        }
    }

    #[test]
    fn test_union_or_logic() {
        let r = Region::Union {
            regions: vec![
                Region::Sphere { center: [0.0, 0.0, 0.0], radius: 1.0 },
                Region::Sphere { center: [3.0, 0.0, 0.0], radius: 1.0 },
            ],
        };
        // In first sphere
        assert!(r.contains(&[0.0, 0.0, 0.0]));
        // In second sphere
        assert!(r.contains(&[3.0, 0.0, 0.0]));
        // In neither
        assert!(!r.contains(&[1.5, 0.0, 0.0]));
    }

    #[test]
    fn test_intersect_and_logic() {
        let r = Region::Intersect {
            regions: vec![
                Region::Sphere { center: [0.0, 0.0, 0.0], radius: 2.0 },
                Region::Sphere { center: [1.0, 0.0, 0.0], radius: 2.0 },
            ],
        };
        // In both spheres
        assert!(r.contains(&[0.5, 0.0, 0.0]));
        // In first only
        assert!(!r.contains(&[-1.5, 0.0, 0.0]));
        // In second only
        assert!(!r.contains(&[2.5, 0.0, 0.0]));
    }

    #[test]
    fn test_union_random_point() {
        let r = Region::Union {
            regions: vec![
                Region::Block { min: [0.0, 0.0, 0.0], max: [1.0, 1.0, 1.0] },
                Region::Block { min: [5.0, 5.0, 5.0], max: [6.0, 6.0, 6.0] },
            ],
        };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let p = r.random_point_inside(&mut rng);
            assert!(r.contains(&p), "random point {:?} not in union", p);
        }
    }

    #[test]
    fn test_intersect_random_point() {
        let r = Region::Intersect {
            regions: vec![
                Region::Block { min: [0.0, 0.0, 0.0], max: [2.0, 2.0, 2.0] },
                Region::Block { min: [1.0, 1.0, 1.0], max: [3.0, 3.0, 3.0] },
            ],
        };
        let mut rng = rand::rng();
        for _ in 0..100 {
            let p = r.random_point_inside(&mut rng);
            assert!(r.contains(&p), "random point {:?} not in intersect", p);
            // Should be in the overlap: [1,1,1] to [2,2,2]
            assert!(p[0] >= 1.0 && p[0] <= 2.0);
            assert!(p[1] >= 1.0 && p[1] <= 2.0);
            assert!(p[2] >= 1.0 && p[2] <= 2.0);
        }
    }

    #[test]
    fn test_union_deserialization() {
        let toml_str = r#"type = "union"
regions = [
    { type = "sphere", center = [0, 0, 0], radius = 1.0 },
    { type = "sphere", center = [2, 0, 0], radius = 1.0 }
]"#;
        let r: Region = toml::from_str(toml_str).unwrap();
        assert!(r.contains(&[0.0, 0.0, 0.0]));
        assert!(r.contains(&[2.0, 0.0, 0.0]));
        assert!(!r.contains(&[4.0, 0.0, 0.0]));
    }

    // ══════════════════════════════════════════════════════════════════════
    // VALIDATION: Region edge cases — boundary points, degenerate regions,
    // nested union/intersect, and overlapping regions.
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_block_boundary_points_are_inside() {
        // All faces, edges, and corners of a block should be "inside" (inclusive)
        let r = Region::Block { min: [0.0, 0.0, 0.0], max: [1.0, 1.0, 1.0] };
        // Faces
        assert!(r.contains(&[0.5, 0.5, 0.0])); // bottom face
        assert!(r.contains(&[0.5, 0.5, 1.0])); // top face
        // Edges
        assert!(r.contains(&[0.0, 0.5, 0.0]));
        assert!(r.contains(&[1.0, 0.5, 1.0]));
        // Corners
        assert!(r.contains(&[0.0, 0.0, 0.0]));
        assert!(r.contains(&[1.0, 1.0, 1.0]));
        assert!(r.contains(&[0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_sphere_boundary_on_surface() {
        let r = Region::Sphere { center: [0.0, 0.0, 0.0], radius: 1.0 };
        // Points exactly on the surface should be inside (<=)
        assert!(r.contains(&[1.0, 0.0, 0.0]));
        assert!(r.contains(&[0.0, 1.0, 0.0]));
        assert!(r.contains(&[0.0, 0.0, 1.0]));
        // Just outside
        assert!(!r.contains(&[1.0 + 1e-10, 0.0, 0.0]));
    }

    #[test]
    fn test_zero_volume_block() {
        // Degenerate block with zero volume (min == max)
        let r = Region::Block { min: [1.0, 2.0, 3.0], max: [1.0, 2.0, 3.0] };
        // Only the exact point should be inside
        assert!(r.contains(&[1.0, 2.0, 3.0]));
        assert!(!r.contains(&[1.0 + 1e-15, 2.0, 3.0]));
    }

    #[test]
    fn test_zero_radius_sphere() {
        // Degenerate sphere with zero radius
        let r = Region::Sphere { center: [0.0, 0.0, 0.0], radius: 0.0 };
        assert!(r.contains(&[0.0, 0.0, 0.0]));
        assert!(!r.contains(&[1e-15, 0.0, 0.0]));
    }

    #[test]
    fn test_nested_union_of_intersections() {
        // Union of two intersections: (A ∩ B) ∪ (C ∩ D)
        let r = Region::Union {
            regions: vec![
                Region::Intersect {
                    regions: vec![
                        Region::Block { min: [0.0, 0.0, 0.0], max: [2.0, 2.0, 2.0] },
                        Region::Sphere { center: [0.0, 0.0, 0.0], radius: 1.5 },
                    ],
                },
                Region::Intersect {
                    regions: vec![
                        Region::Block { min: [3.0, 3.0, 3.0], max: [5.0, 5.0, 5.0] },
                        Region::Sphere { center: [4.0, 4.0, 4.0], radius: 2.0 },
                    ],
                },
            ],
        };
        // Point in first intersection
        assert!(r.contains(&[0.5, 0.5, 0.5]));
        // Point in second intersection
        assert!(r.contains(&[4.0, 4.0, 4.0]));
        // Point in neither
        assert!(!r.contains(&[2.5, 2.5, 2.5]));
    }

    #[test]
    fn test_overlapping_union_regions() {
        // Two overlapping spheres — point in overlap should be inside
        let r = Region::Union {
            regions: vec![
                Region::Sphere { center: [0.0, 0.0, 0.0], radius: 2.0 },
                Region::Sphere { center: [1.0, 0.0, 0.0], radius: 2.0 },
            ],
        };
        // In overlap region
        assert!(r.contains(&[0.5, 0.0, 0.0]));
        // In first only
        assert!(r.contains(&[-1.5, 0.0, 0.0]));
        // In second only
        assert!(r.contains(&[2.5, 0.0, 0.0]));
    }

    #[test]
    fn test_non_overlapping_intersect_is_empty() {
        // Two non-overlapping blocks — intersect should contain nothing
        let r = Region::Intersect {
            regions: vec![
                Region::Block { min: [0.0, 0.0, 0.0], max: [1.0, 1.0, 1.0] },
                Region::Block { min: [2.0, 2.0, 2.0], max: [3.0, 3.0, 3.0] },
            ],
        };
        assert!(!r.contains(&[0.5, 0.5, 0.5]));
        assert!(!r.contains(&[2.5, 2.5, 2.5]));
        assert!(!r.contains(&[1.5, 1.5, 1.5]));
    }

    #[test]
    fn test_cylinder_at_axial_boundaries() {
        let r = Region::Cylinder {
            center: [0.0, 0.0],
            radius: 1.0,
            axis: Axis::Z,
            lo: 0.0,
            hi: 5.0,
        };
        // Exactly at lo
        assert!(r.contains(&[0.0, 0.0, 0.0]));
        // Exactly at hi
        assert!(r.contains(&[0.0, 0.0, 5.0]));
        // Just below lo
        assert!(!r.contains(&[0.0, 0.0, -1e-10]));
        // Just above hi
        assert!(!r.contains(&[0.0, 0.0, 5.0 + 1e-10]));
    }

    #[test]
    fn test_plane_exact_on_surface() {
        let r = Region::Plane {
            point: [0.0, 0.0, 5.0],
            normal: [0.0, 0.0, 1.0],
        };
        // Point exactly on the plane
        assert!(r.contains(&[10.0, -5.0, 5.0]));
        // Point just above (positive side)
        assert!(r.contains(&[0.0, 0.0, 5.0 + 1e-15]));
        // Point just below (negative side)
        assert!(!r.contains(&[0.0, 0.0, 5.0 - 1e-10]));
    }

    #[test]
    fn test_region_deserialization() {
        let toml_str = r#"type = "sphere"
center = [1.0, 2.0, 3.0]
radius = 5.0"#;
        let r: Region = toml::from_str(toml_str).unwrap();
        assert!(r.contains(&[1.0, 2.0, 3.0]));
    }
}
