//! Generic symmetric NxN pair coefficient table for multi-type simulations.

/// Mixing rule for combining per-type parameters into pair parameters.
#[derive(Clone, Debug, PartialEq)]
pub enum MixingRule {
    /// sqrt(a_i * a_j)
    Geometric,
    /// (a_i + a_j) / 2
    Arithmetic,
}

/// Symmetric NxN storage for pair coefficients, indexed by atom type.
///
/// Stores coefficients in a flat row-major `ntypes × ntypes` vector.
/// Setting `(i, j)` automatically sets `(j, i)` for symmetry.
pub struct PairCoeffTable<T: Clone> {
    ntypes: usize,
    coeffs: Vec<T>,
}

impl<T: Clone> PairCoeffTable<T> {
    /// Create a new table for `ntypes` atom types, filled with `default`.
    pub fn new(ntypes: usize, default: T) -> Self {
        PairCoeffTable {
            ntypes,
            coeffs: vec![default; ntypes * ntypes],
        }
    }

    /// Look up the coefficient for atom types `i` and `j`.
    #[inline(always)]
    pub fn get(&self, i: u32, j: u32) -> &T {
        debug_assert!((i as usize) < self.ntypes && (j as usize) < self.ntypes);
        unsafe { self.coeffs.get_unchecked(i as usize * self.ntypes + j as usize) }
    }

    /// Set the coefficient for pair `(i, j)` and `(j, i)`.
    pub fn set(&mut self, i: usize, j: usize, val: T) {
        self.coeffs[i * self.ntypes + j] = val.clone();
        self.coeffs[j * self.ntypes + i] = val;
    }

    /// Number of atom types.
    pub fn ntypes(&self) -> usize {
        self.ntypes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_coeff_table_symmetry() {
        let mut table = PairCoeffTable::new(3, 0.0f64);
        table.set(0, 1, 1.5);
        assert_eq!(*table.get(0, 1), 1.5);
        assert_eq!(*table.get(1, 0), 1.5);
    }

    #[test]
    fn test_pair_coeff_table_diagonal() {
        let mut table = PairCoeffTable::new(2, 0.0f64);
        table.set(0, 0, 3.0);
        table.set(1, 1, 5.0);
        assert_eq!(*table.get(0, 0), 3.0);
        assert_eq!(*table.get(1, 1), 5.0);
    }

    #[test]
    fn test_pair_coeff_table_default() {
        let table = PairCoeffTable::new(2, 42.0f64);
        assert_eq!(*table.get(0, 0), 42.0);
        assert_eq!(*table.get(0, 1), 42.0);
    }
}
