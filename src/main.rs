

use matrix::Matrix;
use rand::Rng;
use ndarray::prelude::*;

trait Layer {
    fn forward_propagation(&self, input_data: &Matrix, output: &mut Matrix) -> Result<(), &'static str> ;
    fn backward_propagation(&mut self, output_error: &mut Matrix, input_data: &mut Matrix, learning_rate: f64) -> Result<(), &'static str> ;
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
    x.powf(2.0)
}

fn mse(y_true :&Matrix, y_pred : &Matrix) -> f64 {
    let mut arr = y_true.clone();
    Matrix::static_sub(&mut arr, y_pred).unwrap();
    Matrix::static_map(&mut arr, power2);
    arr.mean()
}

fn mse_prime(y_true :&Matrix, y_pred : &Matrix) -> Matrix {
    let mut arr = y_true.clone();
    Matrix::static_sub(&mut arr, y_pred).unwrap();
    Matrix::static_scalar_mult(&mut arr, 2.0);
    let size = (y_true.size().0 * y_true.size().1) as f64;
    Matrix::static_scalar_mult(&mut arr, 1.0 / size);
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
            println!("epoch {}| Trained on sample size {} | error={}", i, samples, err);
        }
    }
}

fn main() {
    let mut x_train : Vec<Matrix> = vec![];
    let mut y_train : Vec<Matrix> = vec![];

    x_train.push(Matrix::new_from_vec(vec![vec![0.0, 0.0]]).unwrap());
    x_train.push(Matrix::new_from_vec(vec![vec![0.0, 1.0]]).unwrap());
    x_train.push(Matrix::new_from_vec(vec![vec![0.0, 0.0]]).unwrap());
    x_train.push(Matrix::new_from_vec(vec![vec![1.0, 1.0]]).unwrap());

    y_train.push(Matrix::new_from_vec(vec![vec![0.0]]).unwrap());
    y_train.push(Matrix::new_from_vec(vec![vec![1.0]]).unwrap());
    y_train.push(Matrix::new_from_vec(vec![vec![1.0]]).unwrap());
    y_train.push(Matrix::new_from_vec(vec![vec![0.0]]).unwrap());

    let mut net = Network::new(mse, mse_prime);
    net.add(FCLayer::new(2, 3).unwrap(), ActivationLayer::new(tanh, tanh_prime));
    net.add(FCLayer::new(3, 1).unwrap(), ActivationLayer::new(tanh, tanh_prime));

    net.fit(&x_train, &y_train, 1000, 0.1);

    let out = net.predict(&x_train).unwrap();
    println!("{:?}", out);
}
