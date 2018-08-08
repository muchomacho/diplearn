use std::iter::{FromIterator, Iterator};
use std::ops::{Add, Index, IndexMut, Mul, Sub};
use std::slice::{Iter, IterMut};

/*

    This module defines numeric vector and numeric matrix
    
*/

// ----------------------------------------------------------------------
// implement NumVector
// this vector is equal to column vector

#[derive(Clone, Debug)]
pub struct NumVector {
    len: usize,
    data: Vec<f64>,
}

impl NumVector {
    // construct a new vector
    pub fn new(array: Vec<f64>) -> NumVector {
        // panic if input vector is empty
        assert!(!array.is_empty(), "Input array must have elements");

        NumVector {
            len: array.len(),
            data: array,
        }
    }

    // vector size
    pub fn len(&self) -> usize {
        self.len
    }

    // iterator of immutable element reference
    pub fn iter(&self) -> Iter<f64> {
        self.data.iter()
    }

    // iterator of mutable element reference
    pub fn iter_mut(&mut self) -> IterMut<f64> {
        self.data.iter_mut()
    }

    // convert Numvector into Vec<f64>
    pub fn convert_vec(self) -> Vec<f64> {
        self.data
    }

    // sum
    pub fn sum(&self) -> f64 {
        self.iter().fold(0.0, |acc, &elem| acc + elem)
    }

    // hadamard product(element-wise product)
    pub fn hadamard_prod(&self, rhs: &NumVector) -> NumVector {
        assert_eq!(self.len, rhs.len(), "Two vectors must have the same size");
        NumVector::new((0..self.len).map(|i| self[i] * rhs[i]).collect())
    }

    // dot product
    pub fn dot_prod(&self, rhs: &NumVector) -> f64 {
        assert_eq!(self.len, rhs.len(), "Two vectors must have the same size");
        (0..self.len).fold(0.0, |acc, i| acc + self[i] * rhs[i])
    }

    // abs function
    pub fn abs(&self) -> NumVector {
        NumVector::new(self.iter().map(|&elem| elem.abs()).collect())
    }

    // gradient of abs function
    pub fn abs_grad(&self) -> NumVector {
        NumVector::new(
            self.iter()
                .map(|&elem| match elem {
                    i if i > 0.0 => 1.0,
                    i if i == 0.0 => 0.0,
                    _ => -1.0,
                })
                .collect(),
        )
    }
}

// Index trait
impl Index<usize> for NumVector {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

impl IndexMut<usize> for NumVector {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.data[i]
    }
}

// vector-vector Add trait
impl<'a, 'b> Add<&'a NumVector> for &'b NumVector {
    type Output = NumVector;

    fn add(self, rhs: &NumVector) -> NumVector {
        assert_eq!(self.len(), rhs.len(), "Two vectors must have the same size");
        NumVector::new((0..self.len()).map(|i| self[i] + rhs[i]).collect())
    }
}

// vector-vector Sub trait
impl<'a, 'b> Sub<&'a NumVector> for &'b NumVector {
    type Output = NumVector;

    fn sub(self, rhs: &NumVector) -> NumVector {
        assert_eq!(self.len(), rhs.len(), "Two vectors must have the same size");
        NumVector::new((0..self.len()).map(|i| self[i] - rhs[i]).collect())
    }
}

// vector-scalar mul trait
impl<'a> Mul<f64> for &'a NumVector {
    type Output = NumVector;

    fn mul(self, rhs: f64) -> NumVector {
        NumVector::new(self.iter().map(|&elem| elem * rhs).collect())
    }
}

// -----------------------------------------------------------------------------------
// implement NumMatrix

#[derive(Clone, Debug)]
pub struct NumMatrix {
    row: usize,
    column: usize,
    data: Vec<NumVector>,
}

impl NumMatrix {
    // construct new matrix
    pub fn new(array: Vec<Vec<f64>>) -> NumMatrix {
        // panic if input vector is empty
        assert!(
            !array.is_empty() && !array[0].is_empty(),
            "Input vector must have elements"
        );
        let row = array.len();
        let column = array[0].len();
        for i in 0..row {
            assert_eq!(
                array[i].len(),
                column,
                "Each row vector should have the same number of element"
            );
        }

        let mut matrix: Vec<NumVector> = Vec::new();
        for vec in array.into_iter() {
            matrix.push(NumVector::new(vec));
        }

        NumMatrix {
            row,
            column,
            data: matrix,
        }
    }

    // shape of the matrix
    pub fn shape(&self) -> (usize, usize) {
        (self.row, self.column)
    }

    // iterator of immutable row vector reference
    pub fn iter(&self) -> Iter<NumVector> {
        self.data.iter()
    }

    // iterator of mutable row vector reference
    pub fn iter_mut(&mut self) -> IterMut<NumVector> {
        self.data.iter_mut()
    }

    // transpose of matrix
    pub fn transpose(&self) -> NumMatrix {
        let mut trans = vec![vec![0.0; self.row]; self.column];
        for i in 0..self.row {
            for j in 0..self.column {
                trans[j][i] = self[i][j];
            }
        }

        return NumMatrix::new(trans);
    }

    // sum
    pub fn sum(&self) -> f64 {
        self.iter().fold(0.0, |acc, vec| acc + vec.sum())
    }

    // hadamard product(element-wise product)
    pub fn hadamard_prod(&self, rhs: &NumMatrix) -> NumMatrix {
        assert!(
            self.row == rhs.shape().0 && self.column == rhs.shape().1,
            "Two matrix must have the same shape"
        );
        (0..self.row)
            .map(|i| self[i].hadamard_prod(&rhs[i]))
            .collect()
    }

    // trace of matrix
    pub fn trace(&self) -> f64 {
        assert_eq!(
            self.row, self.column,
            "trace is defined for only square matrix"
        );
        (0..self.row).fold(0.0, |acc, i| acc + self[i][i])
    }

    // re-lu function
    // input: X
    pub fn re_lu(&self) -> NumMatrix {
        let mut result = self.clone();
        for i in 0..result.shape().0 {
            for j in 0..result.shape().1 {
                if result[i][j] < 0.0 {
                    result[i][j] = 0.0;
                }
            }
        }
        return result;
    }

    // gradient of re_lu function
    // input: f(X)
    pub fn re_lu_grad(&self) -> NumMatrix {
        let mut result = self.clone();
        for i in 0..result.shape().0 {
            for j in 0..result.shape().1 {
                if result[i][j] > 0.0 {
                    result[i][j] = 1.0;
                }
            }
        }
        return result;
    }

    // soft-max function
    // input: X
    pub fn soft_max(&self) -> NumMatrix {
        let exp_sum: Vec<f64> = (0..self.shape().0)
            .map(|i| (0..self.shape().1).fold(0.0, |acc, j| acc + self[i][j].exp()))
            .collect();
        NumMatrix::new(
            (0..self.shape().0)
                .map(|i| {
                    (0..self.shape().1)
                        .map(|j| self[i][j].exp() / exp_sum[i])
                        .collect()
                })
                .collect(),
        )
    }

    // sigmoid function
    // f(x) = 1 / (1 + e^-x)
    // input: X
    pub fn sigmoid(&self) -> NumMatrix {
        let mut result = NumMatrix::new(vec![vec![1.0; self.shape().1]; self.shape().0]);
        for i in 0..self.shape().0 {
            for j in 0..self.shape().1 {
                result[i][j] += (-1.0 * self[i][j]).exp();
                result[i][j] = 1.0 / result[i][j];
            }
        }
        return result;
    }

    // gradient of sigmoid function
    // f'(x) = f(x)(1 - f(x))
    // input: f(X)
    pub fn sigmoid_grad(&self) -> NumMatrix {
        self.hadamard_prod(&(&(self * (-1.0)) + 1.0))
    }

    // mean squared error function
    // input: X, Y
    pub fn mse(&self, y: &NumMatrix) -> f64 {
        assert_eq!(
            self.shape(),
            y.shape(),
            "Two matrix must have the same shape"
        );
        let diff = self - y;
        let diff_squared = &diff.transpose() * &diff;
        diff_squared.trace() / self.shape().0 as f64
    }

    // gradient of mean squared error
    // input: X, Y
    pub fn mse_grad(&self, y: &NumMatrix) -> NumMatrix {
        assert_eq!(
            self.shape(),
            y.shape(),
            "Two matrix must have the same shape"
        );
        let diff = self - y;
        let grad = self.sigmoid_grad();
        &diff.hadamard_prod(&grad) * (1.0 / self.shape().0 as f64)
    }

    // abs function
    pub fn abs(&self) -> NumMatrix {
        self.iter().map(|vec| vec.abs()).collect()
    }

    // gradient of abs function
    pub fn abs_grad(&self) -> NumMatrix {
        self.iter().map(|vec| vec.abs_grad()).collect()
    }
}

// Index trait
impl Index<usize> for NumMatrix {
    type Output = NumVector;
    fn index(&self, i: usize) -> &NumVector {
        &self.data[i]
    }
}

impl IndexMut<usize> for NumMatrix {
    fn index_mut(&mut self, i: usize) -> &mut NumVector {
        &mut self.data[i]
    }
}

// Matrix-scalar Add trait
impl<'a> Add<f64> for &'a NumMatrix {
    type Output = NumMatrix;

    fn add(self, rhs: f64) -> NumMatrix {
        NumMatrix::new(
            (0..self.shape().0)
                .map(|i| (0..self.shape().1).map(|j| self[i][j] + rhs).collect())
                .collect(),
        )
    }
}

// Matrix-scalar Sub trait
impl<'a> Sub<f64> for &'a NumMatrix {
    type Output = NumMatrix;

    fn sub(self, rhs: f64) -> NumMatrix {
        NumMatrix::new(
            (0..self.shape().0)
                .map(|i| (0..self.shape().1).map(|j| self[i][j] - rhs).collect())
                .collect(),
        )
    }
}

// Matrix-scalar Mul trait
impl<'a> Mul<f64> for &'a NumMatrix {
    type Output = NumMatrix;

    fn mul(self, rhs: f64) -> NumMatrix {
        self.iter().map(|vec| vec * rhs).collect()
    }
}

// Matrix-Vector Mul trait
impl<'a, 'b> Mul<&'a NumVector> for &'b NumMatrix {
    type Output = NumVector;

    fn mul(self, rhs: &NumVector) -> NumVector {
        assert_eq!(
            self.shape().1,
            rhs.len(),
            "Matrix column size and vector size must be the same"
        );
        NumVector::new(self.iter().map(|vec| vec.dot_prod(rhs)).collect())
    }
}

// Matrix-Matrix Add trait
impl<'a, 'b> Add<&'a NumMatrix> for &'b NumMatrix {
    type Output = NumMatrix;

    fn add(self, rhs: &NumMatrix) -> NumMatrix {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Two matrix must have the same size"
        );

        (0..self.shape().0).map(|i| &self[i] + &rhs[i]).collect()
    }
}

// Matrix-Matrix Sub trait
impl<'a, 'b> Sub<&'a NumMatrix> for &'b NumMatrix {
    type Output = NumMatrix;

    fn sub(self, rhs: &NumMatrix) -> NumMatrix {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Two matrix must have the same size"
        );

        (0..self.shape().0).map(|i| &self[i] - &rhs[i]).collect()
    }
}

// Matrix-Matrix Mul trait
impl<'a, 'b> Mul<&'a NumMatrix> for &'b NumMatrix {
    type Output = NumMatrix;

    fn mul(self, rhs: &NumMatrix) -> NumMatrix {
        assert_eq!(
            self.shape().1,
            rhs.shape().0,
            "The former column size and the latter row size must be the same"
        );

        let mut out = vec![vec![0.0; rhs.shape().1]; self.shape().0];
        for i in 0..self.shape().0 {
            for j in 0..rhs.shape().1 {
                for k in 0..self.shape().1 {
                    out[i][j] += self[i][k] * rhs[k][j];
                }
            }
        }

        return NumMatrix::new(out);
    }
}

// FromIterator<NumVector> for NumMatrix
impl FromIterator<NumVector> for NumMatrix {
    fn from_iter<I: IntoIterator<Item = NumVector>>(iter: I) -> Self {
        let mut matrix: Vec<Vec<f64>> = Vec::new();

        for vec in iter {
            matrix.push(vec.convert_vec());
        }

        return NumMatrix::new(matrix);
    }
}
