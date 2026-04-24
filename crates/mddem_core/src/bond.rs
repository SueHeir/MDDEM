//! Per-atom bond topology storage and 1-2/1-3 pair exclusions.
//!
//! [`BondStore`] is registered as an [`AtomData`] extension by [`BondPlugin`].
//! Each atom stores its bond list; both partners of a bond hold a record
//! (A→B and B→A), enabling local exclusion checks without global lookups.

use std::any::Any;

use sim_app::prelude::*;

use crate::AtomData;

// ── BondEntry & BondStore ────────────────────────────────────────────────────

/// A single bond record: who the partner is, the bond type, and the equilibrium length.
#[derive(Clone, Debug)]
pub struct BondEntry {
    /// Global tag of the bonded partner atom.
    pub partner_tag: u32,
    /// Bond type index (for future coefficient table lookups).
    pub bond_type: u32,
    /// Reference (equilibrium) bond length.
    pub r0: f64,
}

/// Per-atom bond topology storage.
///
/// Each local (and ghost) atom has a `Vec<BondEntry>` listing all bonds it
/// participates in. Both atoms in a bonded pair store the bond (A→B and B→A).
pub struct BondStore {
    /// Per-atom bond lists, indexed by local atom index.
    pub bonds: Vec<Vec<BondEntry>>,
}

impl BondStore {
    pub fn new() -> Self {
        BondStore {
            bonds: Vec::new(),
        }
    }

    /// Returns true if local atom `i` has a bond to the atom with global tag `partner_tag`.
    pub fn has_bond(&self, i: usize, partner_tag: u32) -> bool {
        if i >= self.bonds.len() {
            return false;
        }
        self.bonds[i].iter().any(|b| b.partner_tag == partner_tag)
    }

    /// 1-2 and 1-3 pair exclusion check.
    ///
    /// Returns true if the pair (i, j) should be excluded from contact/pair forces:
    /// - **1-2**: atoms i and j are directly bonded
    /// - **1-3**: atoms i and j share a common bonded neighbor
    ///
    /// `tags` is the global tag array so we can look up `tag[i]` and `tag[j]`.
    pub fn are_excluded(&self, i: usize, j: usize, tags: &[u32]) -> bool {
        if i >= self.bonds.len() || j >= self.bonds.len() {
            return false;
        }

        let tag_j = tags[j];

        // 1-2 exclusion: direct bond
        if self.bonds[i].iter().any(|b| b.partner_tag == tag_j) {
            return true;
        }

        // 1-3 exclusion: shared bonded neighbor
        for bi in &self.bonds[i] {
            for bj in &self.bonds[j] {
                if bi.partner_tag == bj.partner_tag {
                    return true;
                }
            }
        }

        false
    }
}

// ── AtomData implementation ──────────────────────────────────────────────────

impl AtomData for BondStore {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn truncate(&mut self, n: usize) {
        self.bonds.resize_with(n, Vec::new);
        self.bonds.truncate(n);
    }

    fn swap_remove(&mut self, i: usize) {
        if i < self.bonds.len() {
            self.bonds.swap_remove(i);
        }
    }

    fn apply_permutation(&mut self, perm: &[usize], n: usize) {
        let new_bonds: Vec<Vec<BondEntry>> =
            perm.iter().map(|&p| self.bonds[p].clone()).collect();
        self.bonds[..n].clone_from_slice(&new_bonds);
    }

    /// Pack format: `[count, (partner_tag, bond_type, r0) × count]` — 1 + 3×count f64s.
    fn pack(&self, i: usize, buf: &mut Vec<f64>) {
        if i < self.bonds.len() {
            let list = &self.bonds[i];
            buf.push(list.len() as f64);
            for entry in list {
                buf.push(entry.partner_tag as f64);
                buf.push(entry.bond_type as f64);
                buf.push(entry.r0);
            }
        } else {
            buf.push(0.0); // no bonds
        }
    }

    fn unpack(&mut self, buf: &[f64]) -> usize {
        let count = buf[0] as usize;
        let mut list = Vec::with_capacity(count);
        let mut pos = 1;
        for _ in 0..count {
            let partner_tag = buf[pos] as u32;
            let bond_type = buf[pos + 1] as u32;
            let r0 = buf[pos + 2];
            list.push(BondEntry {
                partner_tag,
                bond_type,
                r0,
            });
            pos += 3;
        }
        self.bonds.push(list);
        pos
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

/// Registers [`BondStore`] into the [`AtomDataRegistry`].
///
/// Must be added after [`AtomPlugin`](crate::AtomPlugin).
pub struct BondPlugin;

impl Plugin for BondPlugin {
    fn build(&self, app: &mut App) {
        crate::register_atom_data!(app, BondStore::new());
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store_with_bonds() -> BondStore {
        // 4 atoms: tags 10, 20, 30, 40
        // Bonds: 10-20 (type 1, r0=1.0), 20-30 (type 1, r0=1.5)
        let mut store = BondStore::new();
        // atom 0 (tag 10): bonded to 20
        store.bonds.push(vec![BondEntry { partner_tag: 20, bond_type: 1, r0: 1.0 }]);
        // atom 1 (tag 20): bonded to 10 and 30
        store.bonds.push(vec![
            BondEntry { partner_tag: 10, bond_type: 1, r0: 1.0 },
            BondEntry { partner_tag: 30, bond_type: 1, r0: 1.5 },
        ]);
        // atom 2 (tag 30): bonded to 20
        store.bonds.push(vec![BondEntry { partner_tag: 20, bond_type: 1, r0: 1.5 }]);
        // atom 3 (tag 40): no bonds
        store.bonds.push(Vec::new());
        store
    }

    #[test]
    fn pack_unpack_round_trip() {
        let store = make_store_with_bonds();
        // Pack atom 1 (two bonds)
        let mut buf = Vec::new();
        store.pack(1, &mut buf);
        assert_eq!(buf.len(), 1 + 3 * 2); // count + 2 × (tag, type, r0)

        let mut store2 = BondStore::new();
        let consumed = store2.unpack(&buf);
        assert_eq!(consumed, buf.len());
        assert_eq!(store2.bonds.len(), 1);
        assert_eq!(store2.bonds[0].len(), 2);
        assert_eq!(store2.bonds[0][0].partner_tag, 10);
        assert_eq!(store2.bonds[0][0].bond_type, 1);
        assert!((store2.bonds[0][0].r0 - 1.0).abs() < 1e-15);
        assert_eq!(store2.bonds[0][1].partner_tag, 30);
        assert!((store2.bonds[0][1].r0 - 1.5).abs() < 1e-15);
    }

    #[test]
    fn pack_unpack_empty() {
        let store = make_store_with_bonds();
        let mut buf = Vec::new();
        store.pack(3, &mut buf); // atom 3 has no bonds
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0] as usize, 0);

        let mut store2 = BondStore::new();
        let consumed = store2.unpack(&buf);
        assert_eq!(consumed, 1);
        assert_eq!(store2.bonds[0].len(), 0);
    }

    #[test]
    fn pack_out_of_range() {
        let store = BondStore::new();
        let mut buf = Vec::new();
        store.pack(99, &mut buf);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0] as usize, 0);
    }

    #[test]
    fn has_bond_basic() {
        let store = make_store_with_bonds();
        assert!(store.has_bond(0, 20));
        assert!(!store.has_bond(0, 30));
        assert!(store.has_bond(1, 10));
        assert!(store.has_bond(1, 30));
        assert!(!store.has_bond(3, 10));
    }

    #[test]
    fn exclusion_1_2_direct_bond() {
        let store = make_store_with_bonds();
        let tags: Vec<u32> = vec![10, 20, 30, 40];
        // 10-20 are directly bonded
        assert!(store.are_excluded(0, 1, &tags));
        assert!(store.are_excluded(1, 0, &tags));
    }

    #[test]
    fn exclusion_1_3_shared_neighbor() {
        let store = make_store_with_bonds();
        let tags: Vec<u32> = vec![10, 20, 30, 40];
        // 10 and 30 are NOT directly bonded, but share neighbor 20
        assert!(store.are_excluded(0, 2, &tags));
        assert!(store.are_excluded(2, 0, &tags));
    }

    #[test]
    fn no_exclusion_for_unrelated_atoms() {
        let store = make_store_with_bonds();
        let tags: Vec<u32> = vec![10, 20, 30, 40];
        // 10 and 40 have no bond relationship
        assert!(!store.are_excluded(0, 3, &tags));
        assert!(!store.are_excluded(3, 0, &tags));
        // 30 and 40 have no bond relationship
        assert!(!store.are_excluded(2, 3, &tags));
    }

    #[test]
    fn swap_remove_works() {
        let mut store = make_store_with_bonds();
        assert_eq!(store.bonds.len(), 4);
        // Remove atom 1 (tag 20) — last element (atom 3) takes its place
        store.swap_remove(1);
        assert_eq!(store.bonds.len(), 3);
        // Index 1 should now be atom 3's data (empty bonds)
        assert_eq!(store.bonds[1].len(), 0);
        // Index 0 should still be atom 0's data
        assert_eq!(store.bonds[0][0].partner_tag, 20);
    }

    #[test]
    fn truncate_grows_and_shrinks() {
        let mut store = BondStore::new();
        store.truncate(3);
        assert_eq!(store.bonds.len(), 3);
        // Add some data to first entry
        store.bonds[0].push(BondEntry { partner_tag: 5, bond_type: 0, r0: 2.0 });
        store.truncate(1);
        assert_eq!(store.bonds.len(), 1);
        assert_eq!(store.bonds[0][0].partner_tag, 5);
    }

    /// Simulates one dimension of the MPI exchange: pack atom + its bonds on
    /// "rank A", swap-remove, ship the buffer, unpack onto "rank B". Verifies
    /// that bond records travel with the migrating atom, the sender keeps its
    /// remaining bonds in a consistent index layout, and no f64s are left over
    /// in the buffer.
    #[test]
    fn migration_preserves_bonds_across_ranks() {
        use crate::{Atom, AtomDataRegistry};

        // ── Rank A: two atoms bonded to each other ────────────────────────
        let mut atoms_a = Atom::new();
        atoms_a.push_test_atom(10, [0.0, 0.0, 0.0], 0.5, 1.0);
        atoms_a.push_test_atom(20, [1.0, 0.0, 0.0], 0.5, 1.0);
        atoms_a.nlocal = 2;

        let mut registry_a = AtomDataRegistry::new();
        let mut store_a = BondStore::new();
        store_a.bonds.push(vec![BondEntry { partner_tag: 20, bond_type: 1, r0: 1.0 }]);
        store_a.bonds.push(vec![BondEntry { partner_tag: 10, bond_type: 1, r0: 1.0 }]);
        registry_a.register(store_a);

        // ── Migrate atom 0 (tag 10) to rank B ─────────────────────────────
        let migrate_idx = 0;
        let mut buf = Vec::new();
        atoms_a.pack_exchange(migrate_idx, &mut buf);
        registry_a.pack_all(migrate_idx, &mut buf);
        atoms_a.swap_remove(migrate_idx);
        registry_a.swap_remove_all(migrate_idx);
        atoms_a.nlocal -= 1;

        // Rank A now has only tag 20 (swapped into index 0), and its bond to
        // tag 10 must still be present and at the same index.
        assert_eq!(atoms_a.tag[0], 20);
        let store_a_after = registry_a.expect::<BondStore>("rankA after");
        assert_eq!(store_a_after.bonds.len(), 1);
        assert_eq!(store_a_after.bonds[0][0].partner_tag, 10);
        drop(store_a_after);

        // ── Rank B: starts empty, receives the migrating atom ─────────────
        let mut atoms_b = Atom::new();
        atoms_b.nlocal = 0;
        let mut registry_b = AtomDataRegistry::new();
        registry_b.register(BondStore::new());

        let mut pos = 0;
        pos += atoms_b.unpack_atom(&buf[pos..], false);
        pos += registry_b.unpack_all(&buf[pos..]);
        atoms_b.nlocal = 1;

        assert_eq!(pos, buf.len(), "must consume the entire exchange buffer");
        assert_eq!(atoms_b.tag[0], 10);

        let store_b = registry_b.expect::<BondStore>("rankB");
        assert_eq!(store_b.bonds.len(), 1, "rank B has one atom's bond list");
        assert_eq!(store_b.bonds[0].len(), 1, "migrated atom kept its bond");
        assert_eq!(store_b.bonds[0][0].partner_tag, 20);
        assert_eq!(store_b.bonds[0][0].bond_type, 1);
        assert!((store_b.bonds[0][0].r0 - 1.0).abs() < 1e-15);
    }

    /// Same migration, but with multiple bonds on one atom plus a bond-free
    /// passenger atom — exercises the per-atom bond-count prefix.
    #[test]
    fn migration_preserves_multi_bond_and_empty() {
        use crate::{Atom, AtomDataRegistry};

        let mut atoms = Atom::new();
        atoms.push_test_atom(10, [0.0, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(20, [1.0, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(30, [2.0, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(40, [3.0, 0.0, 0.0], 0.5, 1.0);
        atoms.nlocal = 4;

        let mut registry = AtomDataRegistry::new();
        let mut store = BondStore::new();
        // Atom 0 (tag 10): bonded to 20 and 30
        store.bonds.push(vec![
            BondEntry { partner_tag: 20, bond_type: 1, r0: 1.0 },
            BondEntry { partner_tag: 30, bond_type: 2, r0: 2.0 },
        ]);
        store.bonds.push(vec![BondEntry { partner_tag: 10, bond_type: 1, r0: 1.0 }]);
        store.bonds.push(vec![BondEntry { partner_tag: 10, bond_type: 2, r0: 2.0 }]);
        store.bonds.push(Vec::new()); // tag 40 is unbonded
        registry.register(store);

        // Migrate atoms 0 and 3 together (one with bonds, one without).
        let mut buf = Vec::new();
        for idx in [3usize, 0usize] {
            // Iterate high→low so swap_remove is safe
            atoms.pack_exchange(idx, &mut buf);
            registry.pack_all(idx, &mut buf);
            atoms.swap_remove(idx);
            registry.swap_remove_all(idx);
            atoms.nlocal -= 1;
        }

        // Decode on a fresh "rank B".
        let mut atoms_b = Atom::new();
        let mut registry_b = AtomDataRegistry::new();
        registry_b.register(BondStore::new());
        let mut pos = 0;
        for _ in 0..2 {
            pos += atoms_b.unpack_atom(&buf[pos..], false);
            pos += registry_b.unpack_all(&buf[pos..]);
            atoms_b.nlocal += 1;
        }
        assert_eq!(pos, buf.len());
        assert_eq!(atoms_b.nlocal, 2);

        // The receiving rank's bond lists must match the order of unpack.
        let store_b = registry_b.expect::<BondStore>("rankB");
        assert_eq!(store_b.bonds.len(), 2);
        // First unpack was atom tag 40 (no bonds).
        assert_eq!(atoms_b.tag[0], 40);
        assert!(store_b.bonds[0].is_empty());
        // Second unpack was atom tag 10 (two bonds).
        assert_eq!(atoms_b.tag[1], 10);
        assert_eq!(store_b.bonds[1].len(), 2);
        let partners: Vec<u32> = store_b.bonds[1].iter().map(|b| b.partner_tag).collect();
        assert!(partners.contains(&20));
        assert!(partners.contains(&30));
    }
}
