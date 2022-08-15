use std::fmt::{Debug, Display, Formatter};
use std::iter::{Map, Zip};
use std::ops::{Index, IndexMut, Range};
use rand::Rng;

// behavior for static functions, is that if there is a failure mutated target is returned without changing, transactional
// idea is no memory reallocation, expects correct inputs

pub struct Matrix {
    matrix : Vec<Vec<f64>>,
    transpose : bool,
    size: (usize, usize),
}

impl Matrix {
    pub fn new(size : (usize, usize), fill_value : f64)  -> Result<Matrix, &'static str>{
        let (x, y) = size;
        if x == 0 || y == 0 {
            return Err("Invalid size of zero on initialization of matrix")
        }
        Ok(Matrix {
            matrix: {
                let (x, y) = size;
                (0..x).map(
                    |_|
                        (0..y).map(
                            |_| fill_value
                        ).collect()
                ).collect()
            },
            transpose: false,
            size
        })
    }

    pub fn mean(&self) -> f64 {
        let mut collector = 0.0;
        for row in &self.matrix {
            for i in row {
                collector += i
            }
        }
        collector / (self.size.0 * self.size.1) as f64
    }

    pub fn new_rand(size : (usize, usize), range : Range<f64>)  -> Result<Matrix, &'static str> {
        let (x, y) = size;
        if x == 0 || y == 0 {
            return Err("Invalid size of zero on initialization of matrix")
        }
        Ok(Matrix {
            matrix: {
                let mut rng = rand::thread_rng();
                let (x, y) = size;
                (0..x).map(
                    |_|
                        (0..y).map(
                            |_| rng.gen_range(range.clone())
                        ).collect()
                ).collect()
            },
            transpose: false,
            size
        })
    }

    pub fn new_from_vec(vec: Vec<Vec<f64>>)  -> Result<Matrix, &'static str>{
        let h = vec.len();
        if h == 0 {
            return Err("Invalid size of zero on initialization of matrix")
        }
        let w = vec[0].len();
        if w == 0 {
            return Err("Invalid size of zero on initialization of matrix")
        }
        for v in &vec {
            if w != v.len() {
                return Err("Err on input vector, dimension is not consistent")
            }
        }
        Ok(Matrix {
            matrix: vec,
            transpose: false,
            size: (h, w)
        })
    }

    pub fn static_add(target: &mut Matrix, rhs: &Matrix) -> Result<(), &'static str> {
        if target.size() != rhs.size() {
            return Err("Target size mismatched with rhs");
        }
        let (num_rows, num_cols) = target.size();

        for row in 0..num_rows {
            for col in 0..num_cols {
                target[(row, col)] += rhs[(row, col)];
            }
        }

        Ok(())
    }

    pub fn static_sub(target: &mut Matrix, rhs: &Matrix) -> Result<(), &'static str> {
        if target.size() != rhs.size() {
            return Err("Target size mismatched with rhs");
        }
        let (num_rows, num_cols) = target.size();

        for row in 0..num_rows {
            for col in 0..num_cols {
                target[(row, col)] -= rhs[(row, col)];
            }
        }

        Ok(())
    }

    pub fn static_mult(target: &mut Matrix, rhs: &Matrix) -> Result<(), &'static str> {
        if target.size() != rhs.size() {
            return Err("Target size mismatched with rhs");
        }
        let (num_rows, num_cols) = target.size();

        for row in 0..num_rows {
            for col in 0..num_cols {
                target[(row, col)] *= rhs[(row, col)];
            }
        }

        Ok(())
    }

    pub fn static_div(target: &mut Matrix, rhs: &Matrix) -> Result<(), &'static str> {
        if target.size() != rhs.size() {
            return Err("Target size mismatched with rhs");
        }
        let (num_rows, num_cols) = target.size();

        for row in 0..num_rows {
            for col in 0..num_cols {
                target[(row, col)] /= rhs[(row, col)];
            }
        }

        Ok(())
    }

    pub fn static_scalar_mult(target: &mut Matrix, val :f64) {
        let (num_rows, num_cols) = target.size();

        for row in 0..num_rows {
            for col in 0..num_cols {
                target[(row, col)] *= val;
            }
        }
    }

    pub fn static_map(target: &mut Matrix, f : fn(f64) -> f64) {
        let (num_rows, num_cols) = target.size();

        for row in 0..num_rows {
            for col in 0..num_cols {
                target[(row, col)] = f(target[(row, col)]);
            }
        }
    }

    pub fn static_apply(target: &mut Matrix, init :&Matrix, f : fn(f64) -> f64) -> Result<(), &'static str> {
        if target.size() != init.size() {
            println!("{:?},{:?}", target.size(), init.size());
            return Err("Mismatched matrix sizes for static apply")
        }
        let (num_rows, num_cols) = target.size();

        for row in 0..num_rows {
            for col in 0..num_cols {
                target[(row, col)] = f(init[(row, col)]);
            }
        }
        Ok(())
    }

    pub fn static_dot(target: &mut Matrix, lhs: &Matrix, rhs: &Matrix) -> Result<(), &'static str> {
        let (l_h, l_w) = lhs.size();
        let (r_h, r_w) = rhs.size();
        let (t_h, t_w) = target.size();
        if l_w != r_h {
            return Err("Invalid dot operation, lhs and rhs have mismatched width and height")
        }
        if l_h != t_h || r_w != t_w {
            return Err("Invalid target for dot operation, make sure sides lengths match with target")
        }
        let (num_rows, num_cols) = target.size();

        for row in 0..num_rows {
            for col in 0..num_cols {
                target[(row, col)] = 0.0;
                for i in 0..l_w {
                    target[(row, col)] += lhs[(row, i)] * rhs[(i, col)]
                }
            }
        }

        Ok(())
    }

    pub fn static_transpose(target: &mut Matrix) {
        target.transpose = !target.transpose;
    }

    pub fn size(&self) -> (usize, usize){
        if self.transpose {
            return (self.size.1, self.size.0);
        }
        self.size
    }
    
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if self.transpose {
            return &self.matrix[index.1][index.0];
        }
        &self.matrix[index.0][index.1]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if self.transpose {
            return &mut self.matrix[index.1][index.0];
        }
        &mut self.matrix[index.0][index.1]
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
                    row_s.push_str(format!("{},", self[(row, col)]).as_str())
                } else {
                    row_s.push_str(format!("{}", self[(row, col)]).as_str())
                }
            }
            if row != num_rows - 1 {
                matrix_s.push_str(format!("[{}],\n", row_s).as_str())
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
            matrix: {
                let mut cloned = vec![];
                for row in &self.matrix {
                    cloned.push(row.clone());
                }
                cloned
            },
            transpose: self.transpose,
            size: self.size
        }
    }
}

impl Eq for Matrix {}

#[cfg(test)]
mod tests {
    use crate::{Matrix};

    #[test]
    fn test_print() {
        let mut matrix = Matrix::new_rand((3, 2), 0.0..3.0).unwrap();
        println!("{:?}", matrix);
        Matrix::static_transpose(&mut matrix);
        println!("{:?}", matrix);
    }

    #[test]
    fn test_eq() {
        let mut matrix1 = Matrix::new_from_vec(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]).unwrap();
        let mut matrix2 = Matrix::new_from_vec(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]).unwrap();
        let mut matrix3 = Matrix::new_from_vec(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let mut matrix4 = Matrix::new_from_vec(vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0,6.0]]).unwrap();

        assert_eq!(matrix1, matrix2);
        Matrix::static_transpose(&mut matrix2);
        assert_ne!(matrix1, matrix2);
        assert_ne!(matrix1, matrix3);
        assert_ne!(matrix3, matrix4);
        Matrix::static_transpose(&mut matrix3);
        assert_eq!(matrix4, matrix3);
    }

    #[test]
    fn test_static_add() {
        let mut matrix1 = Matrix::new_from_vec(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let mut matrix2 = Matrix::new_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0,6.0]]).unwrap();
        let matrix3 = Matrix::new_from_vec(vec![vec![2.0, 5.0, 8.0], vec![6.0, 9.0, 12.0]]).unwrap();
        Matrix::static_transpose(&mut matrix2);
        Matrix::static_add(&mut matrix1, &matrix2).unwrap();
        assert_eq!(matrix1, matrix3);
    }

    #[test]
    fn test_static_dot() {
        let matrix1 = Matrix::new_from_vec(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let matrix2 = Matrix::new_from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0,6.0]]).unwrap();
        let mut matrix3 = Matrix::new_from_vec(vec![vec![1.0, 2.0], vec![4.0, 5.0]]).unwrap();
        let matrix4 = Matrix::new_from_vec(vec![vec![22.0, 28.0], vec![49.0, 64.0]]).unwrap();
        Matrix::static_dot(&mut matrix3, &matrix1, &matrix2).unwrap();
        assert_eq!(matrix3, matrix4);
    }

}
