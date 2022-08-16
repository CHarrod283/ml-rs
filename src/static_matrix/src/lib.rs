
struct Matrix {
    arr : Vec<f64>,
    size : (usize, usize),
    transpose : bool
}

impl Matrix {
    fn new(num_rows : usize, num_cols : usize) -> SMatrix {
        SMatrix {
            arr: vec![0.0, num_rows * num_cols],
            size: (num_rows, num_cols),
            transpose: false
        }
    }

    fn add
}



#[cfg(test)]
mod tests {
    use crate::new;

    #[test]
    fn it_works() {
        const b : usize = 1;
        let a = new::<b>(&10);
        println!("{:?}", a)
    }
}
