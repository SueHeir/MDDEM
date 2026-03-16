//! Per-atom angle (3-body) topology storage for bond angle potentials.

use std::any::Any;

use mddem_app::prelude::*;

use crate::AtomData;

// ── AngleEntry & AngleStore ──────────────────────────────────────────────────

/// A single angle record: three atoms forming an angle (i—j—k),
/// stored on the central atom j.
#[derive(Clone, Debug)]
pub struct AngleEntry {
    /// Global tag of the first end atom.
    pub tag_i: u32,
    /// Global tag of the second end atom.
    pub tag_k: u32,
    /// Angle type index (for coefficient table lookups).
    pub angle_type: u32,
}

/// Per-atom angle topology storage.
///
/// Each atom stores a list of angles where it is the **central** atom.
/// For a chain A—B—C, atom B stores the angle entry {tag_i=A, tag_k=C}.
pub struct AngleStore {
    /// Per-atom angle lists, indexed by local atom index.
    pub angles: Vec<Vec<AngleEntry>>,
}

impl AngleStore {
    pub fn new() -> Self {
        AngleStore {
            angles: Vec::new(),
        }
    }
}

// ── AtomData implementation ──────────────────────────────────────────────────

impl AtomData for AngleStore {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn truncate(&mut self, n: usize) {
        self.angles.resize_with(n, Vec::new);
        self.angles.truncate(n);
    }

    fn swap_remove(&mut self, i: usize) {
        if i < self.angles.len() {
            self.angles.swap_remove(i);
        }
    }

    fn apply_permutation(&mut self, perm: &[usize], n: usize) {
        let new_angles: Vec<Vec<AngleEntry>> =
            perm.iter().map(|&p| self.angles[p].clone()).collect();
        self.angles[..n].clone_from_slice(&new_angles);
    }

    /// Pack format: `[count, (tag_i, tag_k, angle_type) × count]` — 1 + 3×count f64s.
    fn pack(&self, i: usize, buf: &mut Vec<f64>) {
        if i < self.angles.len() {
            let list = &self.angles[i];
            buf.push(list.len() as f64);
            for entry in list {
                buf.push(entry.tag_i as f64);
                buf.push(entry.tag_k as f64);
                buf.push(entry.angle_type as f64);
            }
        } else {
            buf.push(0.0);
        }
    }

    fn unpack(&mut self, buf: &[f64]) -> usize {
        let count = buf[0] as usize;
        let mut list = Vec::with_capacity(count);
        let mut pos = 1;
        for _ in 0..count {
            let tag_i = buf[pos] as u32;
            let tag_k = buf[pos + 1] as u32;
            let angle_type = buf[pos + 2] as u32;
            list.push(AngleEntry {
                tag_i,
                tag_k,
                angle_type,
            });
            pos += 3;
        }
        self.angles.push(list);
        pos
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

/// Registers [`AngleStore`] into the [`AtomDataRegistry`].
pub struct AnglePlugin;

impl Plugin for AnglePlugin {
    fn is_unique(&self) -> bool {
        false
    }

    fn build(&self, app: &mut App) {
        crate::register_atom_data!(app, AngleStore::new());
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        let mut store = AngleStore::new();
        store.angles.push(vec![
            AngleEntry { tag_i: 0, tag_k: 2, angle_type: 0 },
            AngleEntry { tag_i: 3, tag_k: 5, angle_type: 1 },
        ]);

        let mut buf = Vec::new();
        store.pack(0, &mut buf);
        assert_eq!(buf.len(), 1 + 3 * 2);

        let mut store2 = AngleStore::new();
        let consumed = store2.unpack(&buf);
        assert_eq!(consumed, buf.len());
        assert_eq!(store2.angles[0].len(), 2);
        assert_eq!(store2.angles[0][0].tag_i, 0);
        assert_eq!(store2.angles[0][0].tag_k, 2);
        assert_eq!(store2.angles[0][1].tag_i, 3);
        assert_eq!(store2.angles[0][1].tag_k, 5);
    }

    #[test]
    fn pack_empty() {
        let store = AngleStore::new();
        let mut buf = Vec::new();
        store.pack(99, &mut buf);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0] as usize, 0);
    }

    #[test]
    fn swap_remove_works() {
        let mut store = AngleStore::new();
        store.angles.push(vec![AngleEntry { tag_i: 0, tag_k: 2, angle_type: 0 }]);
        store.angles.push(Vec::new());
        store.angles.push(vec![AngleEntry { tag_i: 1, tag_k: 3, angle_type: 0 }]);
        store.swap_remove(0);
        assert_eq!(store.angles.len(), 2);
        assert_eq!(store.angles[0][0].tag_i, 1);
    }
}
