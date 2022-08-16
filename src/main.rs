

use matrix::Matrix;
use rand::Rng;
use ndarray::prelude::*;

trait Layer {
    fn forward_propagation(&self, input_data: &Matrix, output: &mut Matrix) -> Result<(), &'static str> ;
    fn backward_propagation(&mut self, derivative_error_wrt_output: &mut Matrix, input: &mut Matrix, learning_rate: f64) -> Result<(), &'static str> ;
}

struct FCLayer  {
    weights : Matrix,
    bias : Matrix,
    weights_error : Matrix,
    input_size : usize,
    output_size : usize,
}

struct ActivationLayer {
    activation : fn(f64) ->f64,
    activation_prime : fn(f64) ->f64,
}

impl FCLayer {
    fn new(input_size : usize, output_size : usize) -> Result<FCLayer, &'static str> {
        Ok(FCLayer {
            weights: Matrix::new_rand((input_size, output_size), -0.5..0.5)?,
            bias: Matrix::new_rand((1, output_size), -0.5..0.5)?,
            weights_error: Matrix::new((input_size, output_size), 0.0)?,
            input_size,
            output_size,
        })
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }
}

impl ActivationLayer {
    fn new(activation : fn(f64) -> f64, activation_prime : fn(f64) -> f64) -> ActivationLayer {
        ActivationLayer {
            activation,
            activation_prime
        }
    }
}

impl Layer for FCLayer {
    fn forward_propagation(&self, input_data: &Matrix, output :&mut Matrix) -> Result<(), &'static str> {
        Matrix::static_dot(output, input_data, &self.weights)?;
        Matrix::static_add(output, &self.bias)?;
        Ok(())
    }


    fn backward_propagation(&mut self, output_error: &mut Matrix, input_data: &mut Matrix, learning_rate: f64) -> Result<(), &'static str> {


        Matrix::static_transpose(input_data);
        Matrix::static_dot(&mut self.weights_error, input_data, output_error)?;
        Matrix::static_transpose(input_data);

        Matrix::static_transpose(&mut self.weights);
        Matrix::static_dot(input_data, output_error, &self.weights)?;
        Matrix::static_transpose(&mut self.weights);

        Matrix::static_scalar_mult(&mut self.weights_error, learning_rate);
        Matrix::static_sub(&mut self.weights, &self.weights_error)?;

        Matrix::static_scalar_mult(output_error, learning_rate);
        Matrix::static_sub(&mut self.bias, output_error)?;
        Ok(())
    }
}

impl Layer for ActivationLayer {
    fn forward_propagation(&self, input_data: &Matrix, output :&mut Matrix) -> Result<(), &'static str> {
        Matrix::static_apply(output, input_data, self.activation)?;
        Ok(())
    }


    fn backward_propagation(&mut self, output_error: &mut Matrix, input_data: &mut Matrix, _learning_rate: f64) -> Result<(), &'static str> {
        Matrix::static_map(input_data, self.activation_prime);
        Matrix::static_mult(input_data, output_error)?;
        Ok(())
    }
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}

fn tanh_prime(x: f64) -> f64 {
    1.0 - (x.tanh()).powf(2.0)
}

fn power2(x: f64) -> f64 {
    x * x
}

fn mse(y_true :&Matrix, y_pred : &Matrix) -> f64 {
    let mut arr = y_true.clone();
    Matrix::static_sub(&mut arr, y_pred).unwrap();
    Matrix::static_map(&mut arr, power2);
    arr.mean()
}

fn mse_prime(y_true :&Matrix, y_pred : &Matrix) -> Matrix {
    let s = y_true.size();
    let size = s.1 * s.0;
    let mul = 2 as f64 / size as f64;
    let mut arr = y_pred.clone();
    Matrix::static_sub(&mut arr, y_true).unwrap();
    Matrix::static_scalar_mult(&mut arr, mul);
    arr
}

struct Network {
    fc_layers : Vec<FCLayer>,
    fc_outs : Vec<Matrix>,
    ac_layers : Vec<ActivationLayer>,
    ac_outs : Vec<Matrix>,
    depth : usize,
    loss : fn(&Matrix, &Matrix) -> f64,
    loss_prime : fn(&Matrix, &Matrix) -> Matrix
}

impl Network {
    fn new(loss : fn(&Matrix, &Matrix) -> f64, loss_prime : fn(&Matrix, &Matrix) -> Matrix) -> Network {
        Network {
            fc_layers: vec![],
            fc_outs: vec![],
            ac_layers: vec![],
            ac_outs: vec![],
            depth: 0,
            loss,
            loss_prime
        }
    }

    fn add_fc(&mut self, l : FCLayer) {
        self.fc_outs.push(Matrix::new((1, l.output_size()), 0.0).unwrap());
        self.fc_layers.push(l);
    }

    fn add_ac(&mut self, l : ActivationLayer, size : usize) {
        self.ac_outs.push(Matrix::new((1, size), 0.0).unwrap());
        self.ac_layers.push(l);
    }

    fn add(&mut self, f : FCLayer, a : ActivationLayer) {
        self.add_ac(a, f.output_size());
        self.add_fc(f);
        self.depth += 1;
    }

    fn predict_individual(&mut self, input_data : &Matrix) -> Result<Matrix, &'static str> {
        for layer in 0..self.depth {
            if layer == 0 {
                // go through fc layer
                self.fc_layers[0].forward_propagation(input_data, &mut self.fc_outs[0])?;
                // go through ac layer
                self.ac_layers[0].forward_propagation(&self.fc_outs[0], &mut self.ac_outs[0])?;
                continue;
            }
            // go through fc layer
            self.fc_layers[layer].forward_propagation(&self.ac_outs[layer - 1], &mut self.fc_outs[layer])?;
            // go through ac layer
            self.ac_layers[layer].forward_propagation(&self.fc_outs[layer], &mut self.ac_outs[layer])?;
        }
        Ok(self.ac_outs.last().expect("Error no added layers to matrix when trying to train").clone())
    }

    fn predict(&mut self, input_data : &[Matrix]) -> Result<Vec<Matrix>, &'static str> {
        let mut results: Vec<Matrix> = Vec::with_capacity(input_data.len());
        for input in input_data {
            results.push(self.predict_individual(input)?);
        }
        Ok(results)
    }

    fn fit(&mut self, x_train : &[Matrix], y_train : &[Matrix], epochs :usize, learning_rate :f64) {
        let samples = x_train.len();

        for i in 0..epochs {
            let mut err = 0.0;
            for j in 0..samples {
                // forward propagate
                let output = self.predict_individual(&x_train[j]).expect("individual pred failed in training");

                // compute loss for report purposes
                err += (self.loss)(&y_train[j], &output);

                let mut output_error = (self.loss_prime)(&y_train[j], &output);

                for layer in (0..self.depth).rev() {

                    // if num layers is 1
                    if self.depth == 1 {
                        // backprop through ac
                        self.ac_layers[layer].backward_propagation(
                            &mut output_error,
                            &mut self.fc_outs[layer], // sets to propagate err
                            learning_rate
                        ).expect("Back Prop failed on ac layer");

                        self.fc_layers[layer].backward_propagation(
                            &mut self.fc_outs[layer],
                            &mut x_train[j].clone(),
                            learning_rate
                        ). expect("back prop failed on fc layer");
                        break;
                    }

                    // if last layer
                    if layer == self.depth - 1 {
                        // backprop through ac
                        self.ac_layers[layer].backward_propagation(
                            &mut output_error,
                            &mut self.fc_outs[layer], // sets to propagate err
                            learning_rate
                        ).expect("Back Prop failed on ac layer");

                        self.fc_layers[layer].backward_propagation(
                            &mut self.fc_outs[layer],
                            &mut self.ac_outs[layer - 1],
                            learning_rate
                        ). expect("back prop failed on fc layer");

                        continue;
                    }

                    // if first layer
                    if layer == 0 {
                        // backprop through ac
                        self.ac_layers[layer].backward_propagation(
                            &mut self.ac_outs[layer],
                            &mut self.fc_outs[layer], // sets to propagate err
                            learning_rate
                        ).expect("Back Prop failed on ac layer");

                        self.fc_layers[layer].backward_propagation(
                            &mut self.fc_outs[layer],
                            &mut x_train[j].clone(),
                            learning_rate
                        ). expect("back prop failed on fc layer");
                        break;
                    }

                    // backprop through ac
                    self.ac_layers[layer].backward_propagation(
                        &mut self.ac_outs[layer],
                        &mut self.fc_outs[layer], // sets to propagate err
                        learning_rate
                    ).expect("Back Prop failed on ac layer");

                    self.fc_layers[layer].backward_propagation(
                        &mut self.fc_outs[layer],
                        &mut self.ac_outs[layer - 1],
                        learning_rate
                    ). expect("back prop failed on fc layer");

                }
            }
            err /= samples as f64;
            println!("epoch {} | Trained on sample size {} | error={}", i, samples, err);
        }
    }
}

#[cfg(test)]
mod tests {
    use matrix::Matrix;
    use crate::{ActivationLayer, FCLayer, Layer, mse, mse_prime, tanh, tanh_prime};

    #[test]
    fn test_fc_forward_propagation() {
        let input_vec = vec![vec![1.0, 0.0]];
        let weights_vec = vec![vec![-0.16201734,  0.05780714, -0.16498584], vec![0.03671397,  0.07464959,  0.04125877]];
        let bias_vec = vec![vec![0.07507767, -0.03768405,  0.07543698]];
        let output_expected_vec = vec![vec![-0.08693967,  0.02012309, -0.08954886]];

        let output_expected = Matrix::new_from_vec(output_expected_vec).expect("err on test expected output");

        let fc_layer = FCLayer {
            weights: Matrix::new_from_vec(weights_vec).expect("err with test weights vec"),
            bias: Matrix::new_from_vec(bias_vec).expect("err with test bias vec"),
            weights_error: Matrix::new((2,3), 0.0).expect("err with test weights vec"),
            input_size: 2,
            output_size: 3
        };
        let mut output = Matrix::new((1, 3), 0.0).expect("err with test output vec");
        fc_layer.forward_propagation(
            &Matrix::new_from_vec(input_vec).expect("err with test input vec"),
            &mut output,

        ).expect("Forward Prop failed");

        assert_eq!(output_expected, output);
    }

    #[test]
    fn test_fc_backward_propagation() {
        let input_vec = vec![vec![-0.52377765, -0.9987279 , -0.59183753]];
        let output_error_vec = vec![vec![ 0.00852554]];
        let weights_vec = vec![vec![1.88864671], vec![-2.21676896], vec![1.00826404]];
        let weights_error_vec = vec![vec![-0.00446549], vec![-0.00851469], vec![-0.00504573]];
        let altered_weights_vec = vec![vec![1.88909326], vec![-2.21591749], vec![1.00876862]];
        let bias_vec = vec![vec![-0.62372669]];
        let altered_bias_vec = vec![vec![-0.62457924]];
        let output_expected_vec = vec![vec![ 0.01610173, -0.01889914,  0.00859599]];

        let mut input = Matrix::new_from_vec(input_vec).expect("err on test expected output");
        let output_expected = Matrix::new_from_vec(output_expected_vec).expect("err on test expected output");
        let mut output_error = Matrix::new_from_vec(output_error_vec).expect("err on test expected output");
        let altered_weights = Matrix::new_from_vec(altered_weights_vec).unwrap();
        let weights_error = Matrix::new_from_vec(weights_error_vec).unwrap();
        let altered_bias = Matrix::new_from_vec(altered_bias_vec).unwrap();


        let mut fc_layer = FCLayer {
            weights: Matrix::new_from_vec(weights_vec).expect("err on test weights vec"),
            bias: Matrix::new_from_vec(bias_vec).expect("err on test bias vec"),
            weights_error: Matrix::new((3, 1), 0.0).expect("err on test weights error vec"),
            input_size: 3,
            output_size: 1
        };

        fc_layer.backward_propagation(
            &mut output_error,
            &mut input,
            0.1
        ).expect("Forward Prop failed");

        assert_eq!(output_expected, input); // passes
        assert_eq!(fc_layer.weights, altered_weights); // passes
        assert_eq!(fc_layer.weights_error, weights_error); // not passes, but technically ok due to difference in implementation
        assert_eq!(fc_layer.bias, altered_bias) // passes
    }

    #[test]
    fn test_ac_forward_propagation() {
        let input_vec = vec![vec![-0.23064345,  1.46985544,  0.47829459]];

        let output_expected_vec = vec![vec![-0.22663884,  0.89954987,  0.44487676]];

        let output_expected = Matrix::new_from_vec(output_expected_vec).expect("err on test expected output");

        let ac_layer = ActivationLayer {
            activation: tanh,
            activation_prime: tanh_prime
        };
        let mut output = Matrix::new((1, 3), 0.0).expect("err with test output vec");
        ac_layer.forward_propagation(
            &Matrix::new_from_vec(input_vec).expect("err with test input vec"),
            &mut output,

        ).expect("Forward Prop failed");

        assert_eq!(output_expected, output);
    }

    #[test]
    fn test_ac_backward_propagation() {
        let input_vec = vec![vec![-0.86212384,  0.69920112,  0.62154165]];
        let output_error_vec = vec![vec![-0.0248505,  -0.0366549 ,  0.02257896]];
        let output_expected_vec = vec![vec![-0.01276577, -0.02328878 , 0.01569406]];

        let mut input = Matrix::new_from_vec(input_vec).expect("err on test expected output");
        let output_expected = Matrix::new_from_vec(output_expected_vec).expect("err on test expected output");
        let mut output_error = Matrix::new_from_vec(output_error_vec).expect("err on test expected output");

        let mut ac_layer = ActivationLayer {
            activation: tanh,
            activation_prime: tanh_prime
        };

        ac_layer.backward_propagation(
            &mut output_error,
            &mut input,
            0.1
        ).expect("Forward Prop failed");

        assert_eq!(output_expected, input);
    }

    #[test]
    fn test_mse() {
        let y_true_vec = vec![vec![1.0]];
        let y_pred_vec = vec![vec![0.81194037]];

        let y_true = Matrix::new_from_vec(y_true_vec).unwrap();
        let y_pred = Matrix::new_from_vec(y_pred_vec).unwrap();
        let out = mse(&y_true, &y_pred);
        assert_eq!(0.035366426165798485, out);
    }

    #[test]
    fn test_mse_prime() {
        let y_true_vec = vec![vec![1.0]];
        let y_pred_vec = vec![vec![0.59020837]];
        let mul = 2;
        let sub = vec![vec![-0.40979163]];
        let output_vec = vec![vec![-0.81958325]];

        let y_true = Matrix::new_from_vec(y_true_vec).unwrap();
        let y_pred = Matrix::new_from_vec(y_pred_vec).unwrap();
        let output = mse_prime(&y_true, &y_pred);
        assert_eq!(Matrix::new_from_vec(output_vec).unwrap(), output);
    }
}

fn main() {
    let mut x_train : Vec<Matrix> = vec![];
    let mut y_train : Vec<Matrix> = vec![];

    x_train.push(Matrix::new_from_vec(vec![vec![0.0, 0.0]]).unwrap());
    x_train.push(Matrix::new_from_vec(vec![vec![0.0, 1.0]]).unwrap());
    x_train.push(Matrix::new_from_vec(vec![vec![1.0, 0.0]]).unwrap());
    x_train.push(Matrix::new_from_vec(vec![vec![1.0, 1.0]]).unwrap());

    y_train.push(Matrix::new_from_vec(vec![vec![0.0]]).unwrap());
    y_train.push(Matrix::new_from_vec(vec![vec![1.0]]).unwrap());
    y_train.push(Matrix::new_from_vec(vec![vec![1.0]]).unwrap());
    y_train.push(Matrix::new_from_vec(vec![vec![0.0]]).unwrap());

    let mut net = Network::new(mse, mse_prime);
    net.add(FCLayer::new(2, 10).unwrap(), ActivationLayer::new(tanh, tanh_prime));
    net.add(FCLayer::new(10, 1).unwrap(), ActivationLayer::new(tanh, tanh_prime));

    net.fit(&x_train, &y_train, 1000, 0.1);

    let out = net.predict(&x_train).unwrap();
    println!("{:?}", out);
}
