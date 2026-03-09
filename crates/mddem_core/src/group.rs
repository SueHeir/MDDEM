//! Atom groups: named subsets selected by type and/or spatial region.

use mddem_app::prelude::*;
use mddem_scheduler::prelude::*;
use serde::Deserialize;

use crate::{Atom, Config};

// ── Config ──────────────────────────────────────────────────────────────────

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
/// Definition of a single atom group from TOML `[[group]]`.
pub struct GroupDef {
    pub name: String,
    #[serde(rename = "type", default)]
    pub atom_types: Option<Vec<u32>>,
    #[serde(default)]
    pub region_x_low: Option<f64>,
    #[serde(default)]
    pub region_x_high: Option<f64>,
    #[serde(default)]
    pub region_y_low: Option<f64>,
    #[serde(default)]
    pub region_y_high: Option<f64>,
    #[serde(default)]
    pub region_z_low: Option<f64>,
    #[serde(default)]
    pub region_z_high: Option<f64>,
}

// ── Group ───────────────────────────────────────────────────────────────────

/// A named group with a boolean mask over local atoms.
pub struct Group {
    pub name: String,
    pub def: GroupDef,
    pub mask: Vec<bool>,
    pub count: usize,
}

// ── GroupRegistry ───────────────────────────────────────────────────────────

/// Registry of all atom groups. Always contains a built-in "all" group.
pub struct GroupRegistry {
    pub groups: Vec<Group>,
}

impl GroupRegistry {
    pub fn new() -> Self {
        GroupRegistry {
            groups: Vec::new(),
        }
    }

    /// Look up a group by name.
    pub fn get(&self, name: &str) -> Option<&Group> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// Look up a group by name, panicking with a helpful message if not found.
    pub fn expect(&self, name: &str) -> &Group {
        self.get(name).unwrap_or_else(|| {
            let available: Vec<&str> = self.groups.iter().map(|g| g.name.as_str()).collect();
            panic!(
                "Group '{}' not found. Available groups: {:?}",
                name, available
            );
        })
    }
}

impl Default for GroupRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Membership evaluation ───────────────────────────────────────────────────

/// Evaluate whether atom `i` matches a group definition (AND-combine all criteria).
fn evaluate_membership(def: &GroupDef, atoms: &Atom, i: usize) -> bool {
    if let Some(ref types) = def.atom_types {
        if !types.contains(&atoms.atom_type[i]) {
            return false;
        }
    }
    if let Some(lo) = def.region_x_low {
        if atoms.pos_x[i] < lo {
            return false;
        }
    }
    if let Some(hi) = def.region_x_high {
        if atoms.pos_x[i] > hi {
            return false;
        }
    }
    if let Some(lo) = def.region_y_low {
        if atoms.pos_y[i] < lo {
            return false;
        }
    }
    if let Some(hi) = def.region_y_high {
        if atoms.pos_y[i] > hi {
            return false;
        }
    }
    if let Some(lo) = def.region_z_low {
        if atoms.pos_z[i] < lo {
            return false;
        }
    }
    if let Some(hi) = def.region_z_high {
        if atoms.pos_z[i] > hi {
            return false;
        }
    }
    true
}

fn rebuild_group_masks(groups: &mut GroupRegistry, atoms: &Atom) {
    let nlocal = atoms.nlocal as usize;
    for group in groups.groups.iter_mut() {
        group.mask.clear();
        group.mask.resize(nlocal, false);
        group.count = 0;
        if group.name == "all" {
            for m in group.mask.iter_mut() {
                *m = true;
            }
            group.count = nlocal;
        } else {
            for i in 0..nlocal {
                if evaluate_membership(&group.def, atoms, i) {
                    group.mask[i] = true;
                    group.count += 1;
                }
            }
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Registers group setup and per-step rebuild systems.
pub struct GroupPlugin;

impl Plugin for GroupPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# Atom groups — named subsets for selective operations.
# [[group]]
# name = "mobile"
# type = [1, 2]              # optional: match atom_type
# region_z_low = 0.0         # optional: spatial bounds (AND-combined)
# region_z_high = 5.0"#,
        )
    }

    fn build(&self, app: &mut App) {
        app.add_resource(GroupRegistry::new())
            .add_setup_system(setup_groups, ScheduleSetupSet::PostSetup)
            .add_update_system(rebuild_groups, ScheduleSet::PreForce);
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

pub fn setup_groups(
    config: Res<Config>,
    atoms: Res<Atom>,
    comm: Res<crate::CommResource>,
    mut groups: ResMut<GroupRegistry>,
) {
    let defs = config.parse_array::<GroupDef>("group");

    // Always start with the built-in "all" group.
    groups.groups.clear();
    groups.groups.push(Group {
        name: "all".to_string(),
        def: GroupDef {
            name: "all".to_string(),
            atom_types: None,
            region_x_low: None,
            region_x_high: None,
            region_y_low: None,
            region_y_high: None,
            region_z_low: None,
            region_z_high: None,
        },
        mask: Vec::new(),
        count: 0,
    });

    for def in defs {
        if def.name == "all" {
            if comm.rank() == 0 {
                eprintln!("WARNING: Cannot redefine built-in group 'all', skipping.");
            }
            continue;
        }
        if comm.rank() == 0 {
            println!("Group '{}': {:?}", def.name, def);
        }
        groups.groups.push(Group {
            name: def.name.clone(),
            def,
            mask: Vec::new(),
            count: 0,
        });
    }

    rebuild_group_masks(&mut groups, &atoms);
}

pub fn rebuild_groups(atoms: Res<Atom>, mut groups: ResMut<GroupRegistry>) {
    rebuild_group_masks(&mut groups, &atoms);
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn make_atom(positions: &[(f64, f64, f64)], types: &[u32]) -> Atom {
        let mut atom = Atom::new();
        for (i, (px, py, pz)) in positions.iter().enumerate() {
            atom.push_test_atom(i as u32, Vector3::new(*px, *py, *pz), 0.5, 1.0);
            atom.atom_type[i] = types[i];
        }
        atom.nlocal = positions.len() as u32;
        atom
    }

    #[test]
    fn test_all_group_always_exists() {
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "all".to_string(),
            def: GroupDef {
                name: "all".to_string(),
                atom_types: None,
                region_x_low: None,
                region_x_high: None,
                region_y_low: None,
                region_y_high: None,
                region_z_low: None,
                region_z_high: None,
            },
            mask: Vec::new(),
            count: 0,
        });
        let atom = make_atom(&[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], &[0, 1]);
        rebuild_group_masks(&mut registry, &atom);
        let all = registry.expect("all");
        assert_eq!(all.count, 2);
        assert!(all.mask.iter().all(|&m| m));
    }

    #[test]
    fn test_type_filter() {
        let atom = make_atom(
            &[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
            &[0, 1, 0],
        );
        let def = GroupDef {
            name: "type0".to_string(),
            atom_types: Some(vec![0]),
            region_x_low: None,
            region_x_high: None,
            region_y_low: None,
            region_y_high: None,
            region_z_low: None,
            region_z_high: None,
        };
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "type0".to_string(),
            def,
            mask: Vec::new(),
            count: 0,
        });
        rebuild_group_masks(&mut registry, &atom);
        let g = registry.expect("type0");
        assert_eq!(g.count, 2);
        assert_eq!(g.mask, vec![true, false, true]);
    }

    #[test]
    fn test_region_filter() {
        let atom = make_atom(
            &[(0.0, 0.0, 1.0), (0.0, 0.0, 3.0), (0.0, 0.0, 5.0)],
            &[0, 0, 0],
        );
        let def = GroupDef {
            name: "bottom".to_string(),
            atom_types: None,
            region_x_low: None,
            region_x_high: None,
            region_y_low: None,
            region_y_high: None,
            region_z_low: Some(0.0),
            region_z_high: Some(4.0),
        };
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "bottom".to_string(),
            def,
            mask: Vec::new(),
            count: 0,
        });
        rebuild_group_masks(&mut registry, &atom);
        let g = registry.expect("bottom");
        assert_eq!(g.count, 2);
        assert_eq!(g.mask, vec![true, true, false]);
    }

    #[test]
    fn test_combined_type_and_region() {
        let atom = make_atom(
            &[(0.0, 0.0, 1.0), (0.0, 0.0, 3.0), (0.0, 0.0, 5.0)],
            &[0, 1, 0],
        );
        let def = GroupDef {
            name: "combo".to_string(),
            atom_types: Some(vec![0]),
            region_x_low: None,
            region_x_high: None,
            region_y_low: None,
            region_y_high: None,
            region_z_low: Some(0.0),
            region_z_high: Some(4.0),
        };
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "combo".to_string(),
            def,
            mask: Vec::new(),
            count: 0,
        });
        rebuild_group_masks(&mut registry, &atom);
        let g = registry.expect("combo");
        assert_eq!(g.count, 1); // only atom 0 (type 0, z=1.0)
        assert_eq!(g.mask, vec![true, false, false]);
    }

    #[test]
    fn test_empty_group() {
        let atom = make_atom(
            &[(0.0, 0.0, 1.0), (0.0, 0.0, 3.0)],
            &[0, 0],
        );
        let def = GroupDef {
            name: "empty".to_string(),
            atom_types: Some(vec![99]),
            region_x_low: None,
            region_x_high: None,
            region_y_low: None,
            region_y_high: None,
            region_z_low: None,
            region_z_high: None,
        };
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "empty".to_string(),
            def,
            mask: Vec::new(),
            count: 0,
        });
        rebuild_group_masks(&mut registry, &atom);
        let g = registry.expect("empty");
        assert_eq!(g.count, 0);
        assert!(g.mask.iter().all(|&m| !m));
    }
}
