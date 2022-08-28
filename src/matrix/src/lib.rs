extern crate core;

use std::fmt::{Debug, Formatter};
use std::iter::{Zip};
use std::ops::{Index, IndexMut, Range};
use rand::Rng;

// this library does not allocate memory. this is intentional



pub struct Matrix {
    matrix : Vec<f64>,
    transpose : bool,
    size: (usize, usize),
}
// read only view of matrix which can transpose it's view
pub struct MatrixView<'a> {
    m : &'a Matrix,
    transpose : bool,
}
// utility structs for iterating through matrix
enum MatrixIter<'a> {
    Normal(&'a Matrix,usize, usize),
    Transpose(&'a Matrix,usize, usize),
}
// trait where you can read a matrix, allows for ambiguous use of matrix and matrixview
pub trait MatrixLook {
    fn get(&self, pos : (usize, usize)) -> f64;

    fn size(&self) -> (usize, usize);
}
// gives stats info about contents of matrix
pub trait MatrixStats {
    fn mean (&self) -> f64;
}
// mathematical operations of matrixies, doesn't allocate memory
pub trait MatrixMath {
    fn inline_add(&mut self, rhs: &impl MatrixLook) -> Result<(), String>;
    fn inline_sub(&mut self, rhs: &impl MatrixLook) -> Result<(), String>;
    fn inline_mult(&mut self, rhs: &impl MatrixLook) -> Result<(), String>;
    fn inline_div(&mut self, rhs: &impl MatrixLook) -> Result<(), String>;
    fn inline_scalar_mult(&mut self, val: f64);
    fn inline_apply(&mut self, f:fn(f64) -> f64);
    fn inline_transpose(&mut self);
    fn target_add(&mut self, lhs: &impl MatrixLook, rhs: &impl MatrixLook) -> Result<(), String>;
    fn target_sub(&mut self, lhs: &impl MatrixLook, rhs: &impl MatrixLook) -> Result<(), String>;
    fn target_mult(&mut self, lhs: &impl MatrixLook, rhs: &impl MatrixLook) -> Result<(), String>;
    fn target_div(&mut self, lhs: &impl MatrixLook, rhs: &impl MatrixLook) -> Result<(), String>;
    fn target_dot(&mut self, lhs: &impl MatrixLook, rhs: &impl MatrixLook) -> Result<(), String>;
    fn target_scalar_mult(&mut self, src: &impl MatrixLook, val: f64)-> Result<(), String>;
    fn target_apply(&mut self, src: &impl MatrixLook, f:fn(f64) -> f64)-> Result<(), String>;
    fn target_transpose(&mut self, src: &impl MatrixLook) -> Result<(), String>;
}

impl<'a> MatrixView<'a> {
    fn of(m : &'a Matrix) -> MatrixView<'a>{
        MatrixView{
            m,
            transpose : false,
        }
    }
    fn transpose(&mut self) {
        self.transpose = ! self.transpose;
    }
}
impl Matrix {
    // constructors
    pub fn new(size : (usize, usize), fill_value : f64)  -> Result<Matrix, String>{
        let (x, y) = size;
        if x == 0 || y == 0 {
            return Err("Invalid size of zero on initialization of matrix".to_string())
        }
        Ok(Matrix {
            matrix: {
                let (x, y) = size;
                (0..x*y).map(
                    |_| fill_value
                ).collect()
            },
            transpose: false,
            size
        })
    }
    pub fn new_rand(size : (usize, usize), range : Range<f64>)  -> Result<Matrix, String> {
        let (x, y) = size;
        if x == 0 || y == 0 {
            return Err("Invalid size of zero on initialization of matrix".to_string())
        }
        Ok(Matrix {
            matrix: {
                let mut rng = rand::thread_rng();
                let (x, y) = size;
                (0..x*y).map(
                    |_| rng.gen_range(range.clone())
                ).collect()
            },
            transpose: false,
            size
        })
    }
    pub fn new_from_vec(vec: &Vec<Vec<f64>>)  -> Result<Matrix, String>{
        let h = vec.len();
        if h == 0 {
            return Err("Invalid size of zero on initialization of matrix".to_string())
        }
        let w = vec[0].len();
        if w == 0 {
            return Err("Invalid size of zero on initialization of matrix".to_string())
        }
        for v in vec {
            if w != v.len() {
                return Err("Err on input vector, dimension is not consistent".to_string())
            }
        }
        let mut vec_matrix = Vec::with_capacity(h * w);
        for v in vec {
            vec_matrix.append(&mut v.clone())
        }
        Ok(Matrix {
            matrix: vec_matrix,
            transpose: false,
            size: (h, w)
        })
    }
    //utility functions
    pub fn size(&self) -> (usize, usize){
        if self.transpose {
            return (self.size.1, self.size.0)
        }
        self.size
    }
    fn iter(&self) -> MatrixIter<'_> {
        if self.transpose {
            return MatrixIter::Transpose(self, 0, self.size.0 * self.size.1)
        }
        MatrixIter::Normal(self, 0, self.size.0 * self.size.1)
    }
}

// common implementations for matrixview and matrix
impl MatrixLook for Matrix {
    fn get(&self, pos: (usize, usize)) -> f64{
        self[pos]
    }
    fn size(&self) -> (usize, usize) {
        self.size()
    }
}
impl<'a> MatrixLook for MatrixView<'a> {
    fn get(&self, pos: (usize, usize)) -> f64{
        if self.transpose {
            return self.m[(pos.1, pos.0)];
        }
        self.m[pos]
    }
    fn size(&self) -> (usize, usize) {
        if self.transpose {
            return (self.m.size().1, self.m.size().0);
        }
        self.m.size()
    }
}
impl MatrixStats for Matrix {
    fn mean(&self) -> f64 {
        let mut collector = 0.0;
        for i in &self.matrix {
            collector += i;
        }
        collector / (self.size.0 * self.size.1) as f64
    }
}
impl<'a> MatrixStats for MatrixView<'a> {
    fn mean(&self) -> f64 {
        self.m.mean()
    }
}

// matrix impl
impl MatrixMath for Matrix {
    fn inline_add(&mut self, rhs: &impl MatrixLook) -> Result<(), String> {
        if self.size() != rhs.size()  {
            return Err(
                format!("Size of target mismatches rhs: ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        rhs.size().0,
                        rhs.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = self.get((row, col)) + rhs.get((row, col));
            }
        }

        Ok(())
    }
    fn inline_sub(&mut self, rhs: &impl MatrixLook) -> Result<(), String> {
        if self.size() != rhs.size()  {
            return Err(
                format!("Size of target mismatches rhs: ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        rhs.size().0,
                        rhs.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = self.get((row, col)) - rhs.get((row, col));
            }
        }

        Ok(())
    }
    fn inline_div(&mut self, rhs: &impl MatrixLook) -> Result<(), String> {
        if self.size() != rhs.size()  {
            return Err(
                format!("Size of target mismatches rhs: ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        rhs.size().0,
                        rhs.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = self.get((row, col)) / rhs.get((row, col));
            }
        }

        Ok(())
    }
    fn inline_mult(&mut self, rhs: &impl MatrixLook) -> Result<(), String> {
        if self.size() != rhs.size()  {
            return Err(
                format!("Size of target mismatches rhs: ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        rhs.size().0,
                        rhs.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = self.get((row, col)) * rhs.get((row, col));
            }
        }

        Ok(())
    }
    fn inline_scalar_mult(&mut self, val: f64) {
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = self.get((row, col)) * val;
            }
        }
    }
    fn inline_apply(&mut self, f: fn(f64)->f64) {
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = f(self.get((row, col)));
            }
        }
    }
    fn inline_transpose(&mut self) {
        self.transpose = !self.transpose;
    }
    fn target_add(&mut self, lhs: &impl MatrixLook, rhs: &impl MatrixLook) -> Result<(), String> {
        if self.size() != lhs.size() || self.size() != rhs.size() {
            return Err(
                format!("Size of target mismatches rhs or lhs: ({}, {}) vs ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        lhs.size().0,
                        lhs.size().1,
                        rhs.size().0,
                        rhs.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = lhs.get((row, col)) + rhs.get((row, col));
            }
        }

        Ok(())
    }
    fn target_sub(&mut self, lhs: &impl MatrixLook, rhs: &impl MatrixLook) -> Result<(), String> {
        if self.size() != lhs.size() || self.size() != rhs.size() {
            return Err(
                format!("Size of target mismatches rhs or lhs: ({}, {}) vs ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        lhs.size().0,
                        lhs.size().1,
                        rhs.size().0,
                        rhs.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = lhs.get((row, col)) - rhs.get((row, col));
            }
        }

        Ok(())
    }
    fn target_div(&mut self, lhs: &impl MatrixLook, rhs: &impl MatrixLook) -> Result<(), String> {
        if self.size() != lhs.size() || self.size() != rhs.size() {
            return Err(
                format!("Size of target mismatches rhs or lhs: ({}, {}) vs ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        lhs.size().0,
                        lhs.size().1,
                        rhs.size().0,
                        rhs.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = lhs.get((row, col)) / rhs.get((row, col));
            }
        }

        Ok(())
    }
    fn target_mult(&mut self, lhs: &impl MatrixLook, rhs: &impl MatrixLook) -> Result<(), String> {
        if self.size() != lhs.size() || self.size() != rhs.size() {
            return Err(
                format!("Size of target mismatches rhs or lhs: ({}, {}) vs ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        lhs.size().0,
                        lhs.size().1,
                        rhs.size().0,
                        rhs.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = lhs.get((row, col)) * rhs.get((row, col));
            }
        }

        Ok(())
    }
    fn target_scalar_mult(&mut self, src: &impl MatrixLook, val: f64) -> Result<(), String> {
        if self.size() != src.size()  {
            return Err(
                format!("Size of target mismatches src: ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        src.size().0,
                        src.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = src.get((row, col)) * val;
            }
        }

        Ok(())
    }
    fn target_apply(&mut self, src: &impl MatrixLook, f: fn(f64) -> f64) -> Result<(), String> {
        if self.size() != src.size()  {
            return Err(
                format!("Size of target mismatches src: ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        src.size().0,
                        src.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = f(src.get((row, col)));
            }
        }

        Ok(())
    }
    fn target_transpose(&mut self, src: &impl MatrixLook) -> Result<(), String> {
        if self.size().0 != src.size().1 || self.size().1 != src.size().0 {
            return Err(
                format!("Size of target mismatches src: ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        src.size().0,
                        src.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = src.get((col, row));
            }
        }

        Ok(())
    }
    fn target_dot(&mut self, lhs: &impl MatrixLook, rhs: &impl MatrixLook) -> Result<(), String> {
        if self.size().0 != lhs.size().0 || self.size().1 != rhs.size().1 || lhs.size().1 != rhs.size().0 {
            return Err(
                format!("Size of target mismatches rhs or lhs: ({}, {}) vs ({}, {}) vs ({}, {})",
                        self.size().0,
                        self.size().1,
                        lhs.size().0,
                        lhs.size().1,
                        rhs.size().0,
                        rhs.size().1
                )
            );
        }
        let (rows, cols) = self.size();
        let is = lhs.size().1;
        for row in 0..rows {
            for col in 0..cols {
                self[(row, col)] = 0.0;
                for i in 0..is {
                    self[(row, col)] += lhs.get((row,i)) * rhs.get((i, col));
                }
            }
        }

        Ok(())
    }
}
impl<'a> Iterator for MatrixIter<'a> {
    type Item = &'a f64;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            MatrixIter::Normal(m, pos, max) => {
                unsafe {
                    let val = m.matrix.get_unchecked(*pos);
                    *pos += 1;
                    if *pos >= *max {
                        return None
                    }
                    Some(val)
                }

            },
            MatrixIter::Transpose(m, pos, max) => {
                unsafe {
                    let val = m.matrix.get_unchecked(transform_index(*pos, m.size.1, m.size.0));
                    *pos += 1;
                    if *pos >= *max {
                        return None
                    }
                    Some(val)
                }
            },
        }
    }
}
impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        let (num_rows, num_cols) = self.size();
        if row >= num_rows {
            panic!("Accessing matrix at out of bounds index, accessed row {} of max {}", row, num_rows - 1 )
        }
        if row >= num_rows {
            panic!("Accessing matrix at out of bounds index, accessed col {} of max {}", col, num_cols - 1 )
        }
        if !self.transpose {
            return &self.matrix[row * num_cols + col]
        }
        &self.matrix[col * num_rows + row]
    }
}
impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        let (num_rows, num_cols) = self.size();
        if row >= num_rows {
            panic!("Accessing matrix at out of bounds index, accessed row {} of max {}", row, num_rows - 1 )
        }
        if row >= num_rows {
            panic!("Accessing matrix at out of bounds index, accessed col {} of max {}", col, num_cols - 1 )
        }
        if !self.transpose {
            return &mut self.matrix[row * num_cols + col]
        }
        &mut self.matrix[col * num_rows + row]
    }
}
impl Debug for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let (num_rows, num_cols) = self.size();

        let mut matrix_s = String::from("");


        for row in 0..num_rows {
            let mut row_s = String::from("");
            for col in 0..num_cols {
                if col != num_cols - 1 {
                    row_s.push_str(format!("{}, ", self[(row, col)]).as_str())
                } else {
                    row_s.push_str(format!("{}", self[(row, col)]).as_str())
                }
            }
            if row != num_rows - 1 {
                matrix_s.push_str(format!("[{}], ", row_s).as_str())
            } else {
                matrix_s.push_str(format!("[{}]", row_s).as_str())
            }
        }


        write!(f, "[{}]", matrix_s)
    }
}
impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.size() != other.size() {
            return false
        }
        let (num_rows, num_cols) = self.size();

        for row in 0..num_rows {
            for col in 0..num_cols {
                if self[(row, col)] != other[(row, col)] {
                    return false
                }
            }
        }
        true
    }
}
impl Clone for Matrix {
    fn clone(&self) -> Self {
        Matrix {
            matrix: self.matrix.clone(),
            transpose: self.transpose,
            size: self.size
        }
    }
}

#[inline]
fn transform_index(i : usize, num_rows : usize, num_cols : usize) -> usize {
    (i % num_rows) * num_cols + i / num_rows // relies on usize truncation
}


#[cfg(test)]
mod tests {

    use crate::{Matrix, MatrixMath};

    #[test]
    fn test_debug_display () {
        let mut matrix1 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]).unwrap();
        let mut matrix2 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let mut matrix3 = Matrix::new_from_vec(&vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]]).unwrap();

        let str1 = format!("{:?}", matrix1);
        let str2 = format!("{:?}", matrix2);
        let str3 = format!("{:?}", matrix3);

        assert_eq!("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", str1);
        assert_eq!("[[1, 2, 3], [4, 5, 6]]", str2);
        assert_eq!("[[1, 2], [4, 5], [7, 8]]", str3);

        matrix1.inline_transpose();
        matrix2.inline_transpose();
        matrix3.inline_transpose();

        let str1 = format!("{:?}", matrix1);
        let str2 = format!("{:?}", matrix2);
        let str3 = format!("{:?}", matrix3);

        assert_eq!("[[1, 4, 7], [2, 5, 8], [3, 6, 9]]", str1);
        assert_eq!("[[1, 4], [2, 5], [3, 6]]", str2);
        assert_eq!("[[1, 4, 7], [2, 5, 8]]", str3);
    }



    #[test]
    fn test_eq() {
        let matrix1 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]).unwrap();
        let mut matrix2 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]).unwrap();
        let mut matrix3 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let matrix4 = Matrix::new_from_vec(&vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0,6.0]]).unwrap();

        assert_eq!(matrix1, matrix2);
        matrix2.inline_transpose();
        assert_ne!(matrix1, matrix2);
        assert_ne!(matrix1, matrix3);
        assert_ne!(matrix3, matrix4);
        matrix3.inline_transpose();
        assert_eq!(matrix4, matrix3);
    }

    #[test]
    fn test_iter() {
        let mut matrix1 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]).unwrap();
        let mut i = 1.0;
        for val in matrix1.iter() {
            assert_eq!(*val, i);
            i += 1.0;
        }
    }

    #[test]
    fn test_target_add() {
        let mut matrix1 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let matrix2 = Matrix::new_from_vec(&vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0,6.0]]).unwrap();
        let mut matrix3 = Matrix::new((3,2), 0_f64).unwrap();
        let expected_val = Matrix::new_from_vec(&vec![vec![2.0, 8.0], vec![4.0, 10.0], vec![6.0,12.0]]).unwrap();

        matrix1.inline_transpose();
        Matrix::target_add(&mut matrix3, &matrix1, &matrix2).expect("Sizes invalid on test");
        assert_eq!(matrix3, expected_val)
    }

    #[test]
    fn test_target_sub() {
        let mut matrix1 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let matrix2 = Matrix::new_from_vec(&vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0,6.0]]).unwrap();
        let mut matrix3 = Matrix::new((3,2), 0_f64).unwrap();
        let expected_val = Matrix::new_from_vec(&vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0,0.0]]).unwrap();

        matrix1.inline_transpose();
        Matrix::target_sub(&mut matrix3, &matrix1, &matrix2).expect("Sizes invalid on test");
        assert_eq!(matrix3, expected_val)
    }

    #[test]
    fn test_target_div() {
        let mut matrix1 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let matrix2 = Matrix::new_from_vec(&vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0,6.0]]).unwrap();
        let mut matrix3 = Matrix::new((3,2), 0_f64).unwrap();
        let expected_val = Matrix::new_from_vec(&vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0,1.0]]).unwrap();

        matrix1.inline_transpose();
        Matrix::target_div(&mut matrix3, &matrix1, &matrix2).expect("Sizes invalid on test");
        assert_eq!(matrix3, expected_val)
    }

    #[test]
    fn test_target_mult() {
        let mut matrix1 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let matrix2 = Matrix::new_from_vec(&vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0,6.0]]).unwrap();
        let mut matrix3 = Matrix::new((3,2), 0_f64).unwrap();
        let expected_val = Matrix::new_from_vec(&vec![vec![1.0, 16.0], vec![4.0, 25.0], vec![9.0,36.0]]).unwrap();

        matrix1.inline_transpose();
        Matrix::target_mult(&mut matrix3, &matrix1, &matrix2).expect("Sizes invalid on test");
        assert_eq!(matrix3, expected_val)
    }

    #[test]
    fn test_target_dot() {
        let mut matrix1 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let matrix2 = Matrix::new_from_vec(&vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0,6.0]]).unwrap();
        let mut matrix3 = Matrix::new((2,2), 0_f64).unwrap();
        let expected_val = Matrix::new_from_vec(&vec![vec![14.0, 32.0], vec![32.0, 77.0]]).unwrap();

        Matrix::target_dot(&mut matrix3, &matrix1, &matrix2).expect("Sizes invalid on test");
        assert_eq!(matrix3, expected_val);

        let mut matrix1 = Matrix::new_from_vec(&vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let mut matrix2 = Matrix::new_from_vec(&vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]]).unwrap();
        let mut matrix3 = Matrix::new((3,3), 0_f64).unwrap();
        let expected_val = Matrix::new_from_vec(&vec![vec![22.0, 29.0, 36.0],vec![27.0, 36.0, 45.0], vec![32.0, 43.0, 54.0]]).unwrap();
        matrix2.inline_transpose();
        Matrix::target_dot(&mut matrix3, &matrix2, &matrix1).expect("Sizes invalid on test");
        assert_eq!(matrix3, expected_val);
    }


}
