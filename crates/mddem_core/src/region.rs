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

/// Result of a closest-point-on-surface query.
///
/// `point` is the nearest point on the region boundary,
/// `normal` is the outward unit normal at that point, and
/// `distance` is the signed distance from the query position to the surface
/// (positive = outside, negative = inside).
#[derive(Debug, Clone, Copy)]
pub struct SurfaceResult {
    pub point: [f64; 3],
    pub normal: [f64; 3],
    pub distance: f64,
}

/// A spatial region primitive. Deserialized from TOML with `type` tag.
///
/// # Examples
/// ```toml
/// region = { type = "block", min = [0, 0, 0], max = [1, 1, 1] }
/// region = { type = "sphere", center = [0, 0, 0], radius = 5.0 }
/// region = { type = "cylinder", center = [0, 0], radius = 1.0, axis = "z", lo = 0.0, hi = 5.0 }
/// region = { type = "cone", center = [0, 0], axis = "z", rad_lo = 0.004, rad_hi = 0.001, lo = 0.0, hi = 0.01 }
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
    /// Truncated cone (frustum) aligned to an axis.
    ///
    /// `rad_lo` is the radius at the `lo` end, `rad_hi` at the `hi` end.
    /// The cone tapers linearly from `rad_lo` to `rad_hi` along the axis.
    Cone {
        center: [f64; 2],
        axis: Axis,
        rad_lo: f64,
        rad_hi: f64,
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
            Region::Cone {
                center,
                axis,
                rad_lo,
                rad_hi,
                lo,
                hi,
            } => {
                let (axial, r0, r1) = match axis {
                    Axis::X => (pos[0], pos[1] - center[0], pos[2] - center[1]),
                    Axis::Y => (pos[1], pos[0] - center[0], pos[2] - center[1]),
                    Axis::Z => (pos[2], pos[0] - center[0], pos[1] - center[1]),
                };
                if axial < *lo || axial > *hi {
                    return false;
                }
                let t = if (hi - lo).abs() > 1e-30 {
                    (axial - lo) / (hi - lo)
                } else {
                    0.5
                };
                let r_at = rad_lo + t * (rad_hi - rad_lo);
                r0 * r0 + r1 * r1 <= r_at * r_at
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
            Region::Cone {
                center,
                axis,
                rad_lo,
                rad_hi,
                lo,
                hi,
            } => {
                // Rejection sampling
                let max_r = rad_lo.max(*rad_hi);
                loop {
                    let u = rng.random_range(-1.0..1.0) * max_r;
                    let v = rng.random_range(-1.0..1.0) * max_r;
                    let axial = rng.random_range(*lo..*hi);
                    let t = if (hi - lo).abs() > 1e-30 {
                        (axial - lo) / (hi - lo)
                    } else {
                        0.5
                    };
                    let r_at = rad_lo + t * (rad_hi - rad_lo);
                    if u * u + v * v <= r_at * r_at {
                        let r0 = center[0] + u;
                        let r1 = center[1] + v;
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

    /// Compute the closest point on the region surface, the outward normal, and the
    /// signed distance from `pos` to the surface (positive = outside, negative = inside).
    pub fn closest_point_on_surface(&self, pos: &[f64; 3]) -> SurfaceResult {
        match self {
            Region::Block { min, max } => closest_point_block(pos, min, max),
            Region::Sphere { center, radius } => closest_point_sphere(pos, center, *radius),
            Region::Cylinder { center, radius, axis, lo, hi } => {
                closest_point_cylinder(pos, center, *radius, axis, *lo, *hi)
            }
            Region::Cone { center, axis, rad_lo, rad_hi, lo, hi } => {
                closest_point_cone(pos, center, axis, *rad_lo, *rad_hi, *lo, *hi)
            }
            Region::Plane { point, normal } => closest_point_plane(pos, point, normal),
            Region::Union { regions } => {
                // Closest surface among all children: minimum absolute distance
                let mut best: Option<SurfaceResult> = None;
                for r in regions {
                    let sr = r.closest_point_on_surface(pos);
                    if best.is_none() || sr.distance.abs() < best.as_ref().unwrap().distance.abs()
                    {
                        best = Some(sr);
                    }
                }
                best.expect("Union must have at least one child region")
            }
            Region::Intersect { regions } => {
                // For intersection, the closest constraining surface is the one
                // with the maximum signed distance (most restrictive boundary).
                // However, for wall contact what matters is the surface the particle
                // would collide with first, so we pick the child whose surface is
                // closest to the point (minimum absolute distance).
                let mut best: Option<SurfaceResult> = None;
                for r in regions {
                    let sr = r.closest_point_on_surface(pos);
                    if best.is_none() || sr.distance.abs() < best.as_ref().unwrap().distance.abs()
                    {
                        best = Some(sr);
                    }
                }
                best.expect("Intersect must have at least one child region")
            }
        }
    }
}

// ── Surface-distance helpers ─────────────────────────────────────────────────

fn closest_point_block(pos: &[f64; 3], min: &[f64; 3], max: &[f64; 3]) -> SurfaceResult {
    // Clamp point to box to find the nearest point on/in the box
    let clamped = [
        pos[0].clamp(min[0], max[0]),
        pos[1].clamp(min[1], max[1]),
        pos[2].clamp(min[2], max[2]),
    ];

    // Check if point is inside the box
    let inside = pos[0] >= min[0]
        && pos[0] <= max[0]
        && pos[1] >= min[1]
        && pos[1] <= max[1]
        && pos[2] >= min[2]
        && pos[2] <= max[2];

    if !inside {
        // Point is outside: closest point on surface is the clamped point
        let dx = pos[0] - clamped[0];
        let dy = pos[1] - clamped[1];
        let dz = pos[2] - clamped[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < 1e-30 {
            // Edge case: point is exactly on surface
            return SurfaceResult {
                point: clamped,
                normal: [0.0, 0.0, 1.0],
                distance: 0.0,
            };
        }
        SurfaceResult {
            point: clamped,
            normal: [dx / dist, dy / dist, dz / dist],
            distance: dist,
        }
    } else {
        // Point is inside: find distance to nearest face
        let distances = [
            pos[0] - min[0], // -X face
            max[0] - pos[0], // +X face
            pos[1] - min[1], // -Y face
            max[1] - pos[1], // +Y face
            pos[2] - min[2], // -Z face
            max[2] - pos[2], // +Z face
        ];
        let normals: [[f64; 3]; 6] = [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ];
        let mut best_idx = 0;
        let mut best_dist = distances[0];
        for i in 1..6 {
            if distances[i] < best_dist {
                best_dist = distances[i];
                best_idx = i;
            }
        }
        // Project to the nearest face
        let normal = normals[best_idx];
        let mut surface_point = *pos;
        match best_idx {
            0 => surface_point[0] = min[0],
            1 => surface_point[0] = max[0],
            2 => surface_point[1] = min[1],
            3 => surface_point[1] = max[1],
            4 => surface_point[2] = min[2],
            _ => surface_point[2] = max[2],
        }
        SurfaceResult {
            point: surface_point,
            normal,
            distance: -best_dist, // negative = inside
        }
    }
}

fn closest_point_sphere(pos: &[f64; 3], center: &[f64; 3], radius: f64) -> SurfaceResult {
    let dx = pos[0] - center[0];
    let dy = pos[1] - center[1];
    let dz = pos[2] - center[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist < 1e-30 {
        // At center: pick arbitrary normal
        return SurfaceResult {
            point: [center[0] + radius, center[1], center[2]],
            normal: [1.0, 0.0, 0.0],
            distance: -radius,
        };
    }

    let inv = 1.0 / dist;
    let nx = dx * inv;
    let ny = dy * inv;
    let nz = dz * inv;

    SurfaceResult {
        point: [center[0] + nx * radius, center[1] + ny * radius, center[2] + nz * radius],
        normal: [nx, ny, nz],
        distance: dist - radius,
    }
}

/// Helper: decompose a 3D position into (axial, d0, d1) for a given axis,
/// where d0/d1 are offsets from the 2D center.
#[inline]
fn decompose_axis(pos: &[f64; 3], center: &[f64; 2], axis: &Axis) -> (f64, f64, f64) {
    match axis {
        Axis::X => (pos[0], pos[1] - center[0], pos[2] - center[1]),
        Axis::Y => (pos[1], pos[0] - center[0], pos[2] - center[1]),
        Axis::Z => (pos[2], pos[0] - center[0], pos[1] - center[1]),
    }
}

/// Helper: compose a 3D point from (axial, r0_world, r1_world) for a given axis.
#[inline]
fn compose_axis(axial: f64, r0_world: f64, r1_world: f64, center: &[f64; 2], axis: &Axis) -> [f64; 3] {
    match axis {
        Axis::X => [axial, center[0] + r0_world, center[1] + r1_world],
        Axis::Y => [center[0] + r0_world, axial, center[1] + r1_world],
        Axis::Z => [center[0] + r0_world, center[1] + r1_world, axial],
    }
}

/// Helper: compose a 3D normal from (axial_component, n0, n1) for a given axis.
#[inline]
fn compose_normal(axial: f64, n0: f64, n1: f64, axis: &Axis) -> [f64; 3] {
    match axis {
        Axis::X => [axial, n0, n1],
        Axis::Y => [n0, axial, n1],
        Axis::Z => [n0, n1, axial],
    }
}

fn closest_point_cylinder(
    pos: &[f64; 3],
    center: &[f64; 2],
    radius: f64,
    axis: &Axis,
    lo: f64,
    hi: f64,
) -> SurfaceResult {
    let (axial, d0, d1) = decompose_axis(pos, center, axis);
    let radial_dist = (d0 * d0 + d1 * d1).sqrt();

    // Clamp axial to [lo, hi]
    let axial_clamped = axial.clamp(lo, hi);
    let axial_inside = axial >= lo && axial <= hi;

    // Candidate 1: curved surface (project radially to radius at clamped axial position)
    let curved_dist;
    let curved_point;
    let curved_normal;
    if radial_dist < 1e-30 {
        // On axis: pick arbitrary radial direction
        curved_point = compose_axis(axial_clamped, radius, 0.0, center, axis);
        curved_normal = compose_normal(0.0, 1.0, 0.0, axis);
        curved_dist = if axial_inside {
            -(radius) // inside, distance to curved wall
        } else {
            let da = if axial < lo { lo - axial } else { axial - hi };
            (radius * radius + da * da).sqrt()
        };
    } else {
        let inv_r = 1.0 / radial_dist;
        let n0 = d0 * inv_r;
        let n1 = d1 * inv_r;
        curved_point = compose_axis(axial_clamped, n0 * radius, n1 * radius, center, axis);
        curved_normal = compose_normal(0.0, n0, n1, axis);
        if axial_inside {
            curved_dist = radial_dist - radius; // positive = outside
        } else {
            // Outside axial range: Euclidean distance to nearest rim
            let da = if axial < lo { lo - axial } else { axial - hi };
            let dr = radial_dist - radius;
            curved_dist = (dr * dr + da * da).sqrt() * if dr < 0.0 && da == 0.0 { -1.0 } else { 1.0 };
        }
    }

    // Candidate 2: lo end cap (flat face at axial = lo)
    let lo_point;
    let lo_normal;
    let lo_dist;
    {
        let cap_r = if radial_dist <= radius { radial_dist } else { radius };
        let (c0, c1) = if radial_dist < 1e-30 {
            (0.0, 0.0)
        } else {
            let inv_r = 1.0 / radial_dist;
            (d0 * inv_r * cap_r, d1 * inv_r * cap_r)
        };
        lo_point = compose_axis(lo, c0, c1, center, axis);
        lo_normal = compose_normal(-1.0, 0.0, 0.0, axis);
        let dx = pos[0] - lo_point[0];
        let dy = pos[1] - lo_point[1];
        let dz = pos[2] - lo_point[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        lo_dist = if axial < lo || (axial_inside && radial_dist <= radius) {
            if axial <= lo { lo - axial } else { d }
        } else {
            d
        };
    }

    // Candidate 3: hi end cap
    let hi_point;
    let hi_normal;
    let hi_dist;
    {
        let cap_r = if radial_dist <= radius { radial_dist } else { radius };
        let (c0, c1) = if radial_dist < 1e-30 {
            (0.0, 0.0)
        } else {
            let inv_r = 1.0 / radial_dist;
            (d0 * inv_r * cap_r, d1 * inv_r * cap_r)
        };
        hi_point = compose_axis(hi, c0, c1, center, axis);
        hi_normal = compose_normal(1.0, 0.0, 0.0, axis);
        let dx = pos[0] - hi_point[0];
        let dy = pos[1] - hi_point[1];
        let dz = pos[2] - hi_point[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        hi_dist = if axial > hi || (axial_inside && radial_dist <= radius) {
            if axial >= hi { axial - hi } else { d }
        } else {
            d
        };
    }

    // If point is inside the cylinder, all distances should be negative
    let inside = axial_inside && radial_dist <= radius;

    if inside {
        // Find the closest face (minimum absolute distance)
        let d_curved = (radius - radial_dist).abs();
        let d_lo = axial - lo;
        let d_hi = hi - axial;

        if d_curved <= d_lo && d_curved <= d_hi {
            SurfaceResult {
                point: curved_point,
                normal: curved_normal,
                distance: radial_dist - radius, // negative
            }
        } else if d_lo <= d_hi {
            SurfaceResult {
                point: lo_point,
                normal: lo_normal,
                distance: -(axial - lo), // negative
            }
        } else {
            SurfaceResult {
                point: hi_point,
                normal: hi_normal,
                distance: -(hi - axial), // negative
            }
        }
    } else {
        // Outside: pick closest candidate by Euclidean distance
        let candidates = [
            (curved_dist.abs(), curved_point, curved_normal, curved_dist),
            (lo_dist, lo_point, lo_normal, lo_dist),
            (hi_dist, hi_point, hi_normal, hi_dist),
        ];
        let mut best = 0;
        for i in 1..3 {
            if candidates[i].0 < candidates[best].0 {
                best = i;
            }
        }
        let (_, point, normal, _) = candidates[best];
        // Compute actual signed distance
        let dx = pos[0] - point[0];
        let dy = pos[1] - point[1];
        let dz = pos[2] - point[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < 1e-30 {
            SurfaceResult {
                point,
                normal,
                distance: 0.0,
            }
        } else {
            SurfaceResult {
                point,
                normal: [dx / dist, dy / dist, dz / dist],
                distance: dist,
            }
        }
    }
}

fn closest_point_cone(
    pos: &[f64; 3],
    center: &[f64; 2],
    axis: &Axis,
    rad_lo: f64,
    rad_hi: f64,
    lo: f64,
    hi: f64,
) -> SurfaceResult {
    let (axial, d0, d1) = decompose_axis(pos, center, axis);
    let radial_dist = (d0 * d0 + d1 * d1).sqrt();
    let height = hi - lo;

    // Radius at a given axial position
    let radius_at = |a: f64| -> f64 {
        if height.abs() < 1e-30 {
            (rad_lo + rad_hi) * 0.5
        } else {
            let t = ((a - lo) / height).clamp(0.0, 1.0);
            rad_lo + t * (rad_hi - rad_lo)
        }
    };

    // Radial direction
    let (n0, n1) = if radial_dist < 1e-30 {
        (1.0, 0.0) // arbitrary
    } else {
        (d0 / radial_dist, d1 / radial_dist)
    };

    // Check if inside
    let axial_inside = axial >= lo && axial <= hi;
    let r_at_pos = radius_at(axial);
    let inside = axial_inside && radial_dist <= r_at_pos;

    // Candidate 1: lateral (conical) surface
    // Project point onto the cone surface in the (axial, radial) 2D space.
    // The cone surface is a line from (lo, rad_lo) to (hi, rad_hi) in (axial, radial) space.
    // We find the closest point on this line segment.
    let cone_result = {
        let la = lo;
        let ra = rad_lo;
        let lb = hi;
        let rb = rad_hi;
        // Line segment: P(t) = (la + t*(lb-la), ra + t*(rb-ra)) for t in [0,1]
        let seg_da = lb - la;
        let seg_dr = rb - ra;
        let seg_len_sq = seg_da * seg_da + seg_dr * seg_dr;
        let t = if seg_len_sq < 1e-30 {
            0.5
        } else {
            let t_raw = ((axial - la) * seg_da + (radial_dist - ra) * seg_dr) / seg_len_sq;
            t_raw.clamp(0.0, 1.0)
        };
        let proj_axial = la + t * seg_da;
        let proj_radial = ra + t * seg_dr;

        // Distance in (axial, radial) space
        let da = axial - proj_axial;
        let dr = radial_dist - proj_radial;
        let dist_2d = (da * da + dr * dr).sqrt();

        // 3D surface point
        let surf_point = compose_axis(proj_axial, n0 * proj_radial, n1 * proj_radial, center, axis);

        // Outward normal: perpendicular to cone surface pointing outward
        // In (axial, radial) space, the cone tangent is (seg_da, seg_dr).
        // The outward normal is (-seg_dr, seg_da) normalized (points away from axis).
        let norm_a = -seg_dr;
        let norm_r = seg_da;
        let norm_len = (norm_a * norm_a + norm_r * norm_r).sqrt();
        let (na, nr) = if norm_len > 1e-30 {
            (norm_a / norm_len, norm_r / norm_len)
        } else {
            (0.0, 1.0)
        };
        let normal_3d = compose_normal(na, nr * n0, nr * n1, axis);

        // Signed distance: positive if outside, negative if inside
        let signed = if inside { -dist_2d } else { dist_2d };
        (surf_point, normal_3d, signed, dist_2d)
    };

    // Candidate 2: lo end cap
    let lo_cap = {
        let cap_r = rad_lo.min(radial_dist.max(0.0)).min(rad_lo);
        let surf_point = compose_axis(lo, n0 * cap_r, n1 * cap_r, center, axis);
        let dx = pos[0] - surf_point[0];
        let dy = pos[1] - surf_point[1];
        let dz = pos[2] - surf_point[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let normal = compose_normal(-1.0, 0.0, 0.0, axis);
        (surf_point, normal, dist)
    };

    // Candidate 3: hi end cap
    let hi_cap = {
        let cap_r = rad_hi.min(radial_dist.max(0.0)).min(rad_hi);
        let surf_point = compose_axis(hi, n0 * cap_r, n1 * cap_r, center, axis);
        let dx = pos[0] - surf_point[0];
        let dy = pos[1] - surf_point[1];
        let dz = pos[2] - surf_point[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let normal = compose_normal(1.0, 0.0, 0.0, axis);
        (surf_point, normal, dist)
    };

    if inside {
        // Pick closest face
        let d_cone = cone_result.3;
        let d_lo = axial - lo;
        let d_hi = hi - axial;

        if d_cone <= d_lo && d_cone <= d_hi {
            SurfaceResult {
                point: cone_result.0,
                normal: cone_result.1,
                distance: cone_result.2,
            }
        } else if d_lo <= d_hi {
            SurfaceResult {
                point: lo_cap.0,
                normal: lo_cap.1,
                distance: -d_lo,
            }
        } else {
            SurfaceResult {
                point: hi_cap.0,
                normal: hi_cap.1,
                distance: -d_hi,
            }
        }
    } else {
        // Outside: pick closest by Euclidean distance
        let candidates = [
            (cone_result.3, cone_result.0, cone_result.1),
            (lo_cap.2, lo_cap.0, lo_cap.1),
            (hi_cap.2, hi_cap.0, hi_cap.1),
        ];
        let mut best = 0;
        for i in 1..3 {
            if candidates[i].0 < candidates[best].0 {
                best = i;
            }
        }
        let (_, point, _normal) = candidates[best];
        let dx = pos[0] - point[0];
        let dy = pos[1] - point[1];
        let dz = pos[2] - point[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < 1e-30 {
            SurfaceResult {
                point,
                normal: _normal,
                distance: 0.0,
            }
        } else {
            SurfaceResult {
                point,
                normal: [dx / dist, dy / dist, dz / dist],
                distance: dist,
            }
        }
    }
}

fn closest_point_plane(pos: &[f64; 3], point: &[f64; 3], normal: &[f64; 3]) -> SurfaceResult {
    // Normalize normal
    let mag = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    let (nx, ny, nz) = if mag > 1e-30 {
        (normal[0] / mag, normal[1] / mag, normal[2] / mag)
    } else {
        (0.0, 0.0, 1.0)
    };

    let dx = pos[0] - point[0];
    let dy = pos[1] - point[1];
    let dz = pos[2] - point[2];
    let dist = dx * nx + dy * ny + dz * nz;

    SurfaceResult {
        point: [pos[0] - dist * nx, pos[1] - dist * ny, pos[2] - dist * nz],
        normal: [nx, ny, nz],
        distance: dist,
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
    fn test_cone_contains() {
        let r = Region::Cone {
            center: [0.0, 0.0],
            axis: Axis::Z,
            rad_lo: 2.0,
            rad_hi: 1.0,
            lo: 0.0,
            hi: 10.0,
        };
        // Center of cone at z=0 (radius=2)
        assert!(r.contains(&[0.0, 0.0, 0.0]));
        // At z=5, radius should be 1.5
        assert!(r.contains(&[1.0, 0.0, 5.0]));
        assert!(!r.contains(&[1.6, 0.0, 5.0]));
        // At z=10, radius should be 1.0
        assert!(r.contains(&[0.5, 0.5, 10.0]));
        assert!(!r.contains(&[1.0, 1.0, 10.0]));
        // Outside axial bounds
        assert!(!r.contains(&[0.0, 0.0, -0.1]));
        assert!(!r.contains(&[0.0, 0.0, 10.1]));
    }

    #[test]
    fn test_cone_random_point() {
        let r = Region::Cone {
            center: [1.0, 2.0],
            axis: Axis::Z,
            rad_lo: 3.0,
            rad_hi: 1.0,
            lo: 0.0,
            hi: 5.0,
        };
        let mut rng = rand::rng();
        for _ in 0..200 {
            let p = r.random_point_inside(&mut rng);
            assert!(r.contains(&p), "random point {:?} not in cone", p);
        }
    }

    #[test]
    fn test_cone_deserialization() {
        let toml_str = r#"type = "cone"
center = [0, 0]
axis = "z"
rad_lo = 0.004
rad_hi = 0.001
lo = 0.0
hi = 0.01"#;
        let r: Region = toml::from_str(toml_str).unwrap();
        assert!(r.contains(&[0.0, 0.0, 0.005]));
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

    // ── Surface distance tests ──────────────────────────────────────────────

    #[test]
    fn test_sphere_surface_outside() {
        let r = Region::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let sr = r.closest_point_on_surface(&[2.0, 0.0, 0.0]);
        assert!((sr.distance - 1.0).abs() < 1e-10, "dist={}", sr.distance);
        assert!((sr.point[0] - 1.0).abs() < 1e-10);
        assert!((sr.normal[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_surface_inside() {
        let r = Region::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let sr = r.closest_point_on_surface(&[0.5, 0.0, 0.0]);
        assert!((sr.distance - (-0.5)).abs() < 1e-10, "dist={}", sr.distance);
        assert!((sr.point[0] - 1.0).abs() < 1e-10);
        assert!((sr.normal[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_block_surface_inside() {
        let r = Region::Block {
            min: [0.0, 0.0, 0.0],
            max: [2.0, 2.0, 2.0],
        };
        // Point near the +z face
        let sr = r.closest_point_on_surface(&[1.0, 1.0, 1.8]);
        assert!(sr.distance < 0.0, "should be inside");
        assert!((sr.distance - (-0.2)).abs() < 1e-10, "dist={}", sr.distance);
        assert!((sr.normal[2] - 1.0).abs() < 1e-10, "normal should be +z");
    }

    #[test]
    fn test_block_surface_outside() {
        let r = Region::Block {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        let sr = r.closest_point_on_surface(&[0.5, 0.5, 1.5]);
        assert!((sr.distance - 0.5).abs() < 1e-10, "dist={}", sr.distance);
        assert!((sr.normal[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_plane_surface() {
        let r = Region::Plane {
            point: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        };
        let sr = r.closest_point_on_surface(&[1.0, 2.0, 3.0]);
        assert!((sr.distance - 3.0).abs() < 1e-10);
        assert!((sr.point[2]).abs() < 1e-10);
        assert!((sr.normal[2] - 1.0).abs() < 1e-10);

        let sr2 = r.closest_point_on_surface(&[1.0, 2.0, -1.0]);
        assert!((sr2.distance - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cylinder_surface_inside_radial() {
        let r = Region::Cylinder {
            center: [0.0, 0.0],
            radius: 1.0,
            axis: Axis::Z,
            lo: 0.0,
            hi: 10.0,
        };
        // Point inside, closer to curved surface than end caps
        let sr = r.closest_point_on_surface(&[0.8, 0.0, 5.0]);
        assert!(sr.distance < 0.0, "should be inside");
        assert!((sr.distance - (-0.2)).abs() < 1e-10, "dist={}", sr.distance);
        // Normal should point radially outward (+x)
        assert!((sr.normal[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cone_surface_inside() {
        let r = Region::Cone {
            center: [0.0, 0.0],
            axis: Axis::Z,
            rad_lo: 2.0,
            rad_hi: 2.0, // cylinder-like cone
            lo: 0.0,
            hi: 10.0,
        };
        // Should behave like a cylinder for constant radius
        let sr = r.closest_point_on_surface(&[1.8, 0.0, 5.0]);
        assert!(sr.distance < 0.0, "should be inside, got {}", sr.distance);
    }

    #[test]
    fn test_union_surface() {
        let r = Region::Union {
            regions: vec![
                Region::Sphere { center: [0.0, 0.0, 0.0], radius: 1.0 },
                Region::Sphere { center: [5.0, 0.0, 0.0], radius: 1.0 },
            ],
        };
        // Closer to first sphere
        let sr = r.closest_point_on_surface(&[0.5, 0.0, 0.0]);
        assert!(sr.distance < 0.0, "inside first sphere");
    }
}
