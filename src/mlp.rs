/*

    This module defines multi-layer perceptron
    Only regression is currently available

*/

extern crate rand;

use numrs::*;
use rand::distributions::{Distribution, Normal};
use std::f64;

static CONVERGE: f64 = 0.000000001;

pub struct MLP {
    layer_num: usize,
    dims: Vec<usize>,
    weight: Vec<NumMatrix>,
    bias: Vec<NumVector>,
}

impl MLP {
    // create MLP learner
    pub fn new(dims: Vec<usize>) -> MLP {
        assert!(dims.len() > 2, "At least one hidden layer is needed");

        // initialization
        // initialize weight matrix with N(0, 0.01) value
        let normal_dist = Normal::new(0.0, 0.1);
        let mut rng = rand::thread_rng();
        let mut w = Vec::new();
        let mut b = Vec::new();
        for i in 0..(dims.len() - 1) {
            let mut matrix = vec![vec![0.0; dims[i + 1]]; dims[i]];
            for j in 0..dims[i] {
                for k in 0..dims[i + 1] {
                    matrix[j][k] = normal_dist.sample(&mut rng);
                }
            }
            let mut vec = vec![0.0; dims[i + 1]];
            w.push(NumMatrix::new(matrix));
            b.push(NumVector::new(vec));
        }

        MLP {
            layer_num: dims.len(),
            dims: dims,
            weight: w,
            bias: b,
        }
    }

    /* 
    train with dataset
    
    Parameter
    x: NumVector (N * D)
    train data
    N = sample size
    D = input dim

    y: Vector
    true values for train data x

    max_iter: usize
    maximum learning loop number

    learning_rate: float
    coefficient of gradient for updating parameter

    regularization: &str
    regularization method
    only L1 reguralization is currently available

    lambda: float
    coefficient of regularization unit

    */
    pub fn train(
        &mut self,
        x: &NumMatrix,
        y: &NumMatrix,
        max_iter: usize,
        learning_rate: f64,
        regularization: &str,
        lambda: f64,
    ) {
        let sample_num = x.shape().0;

        assert_eq!(
            x.shape().0,
            y.shape().0,
            "Train data and label must have the same size"
        );
        assert_eq!(x.shape().1, self.dims[0], "Input vector size is different");
        assert_eq!(
            y.shape().1,
            self.dims[self.layer_num - 1],
            "Output vector size is different"
        );

        // node value matrix of each layer
        let mut node_vals: Vec<NumMatrix>;
        // error gradient value matrix of each layer
        let mut error_grad: Vec<NumMatrix>;

        // training loop
        // i = 0 : Input layer, i = l - 1 : Output layer
        // node_vals[i] = node values at i-th layer
        // error_grad[i] = error gradient at i-th layer
        // weight[i], bias[i] = parameter from i-th layer to (i + 1)-th layer
        let mut old_val = f64::MAX;
        println!("Epoch\tObjective function value");
        for t in 0..max_iter {
            node_vals = self.forward_prop(x);
            error_grad = self.back_prop(y, &node_vals);

            // if converged, finish training
            // F = mse + lambda * sum(weight.abs) + lambda * sum(bias.abs)
            let mut obj_func_val = node_vals[self.layer_num - 1].mse(y);
            for i in 0..self.layer_num - 1 {
                obj_func_val += (self.weight[i].abs().sum() + self.bias[i].abs().sum()) * lambda;
            }
            if (obj_func_val - old_val).abs() < CONVERGE {
                println!("Error value converged\nFinish training");
                break;
            }
            old_val = obj_func_val;

            // update parameter
            // W = W - eps * W_grad
            // b = b - eps * b_grad
            for i in 0..(self.layer_num - 1) {
                let mut w_grad = &node_vals[i].transpose() * &error_grad[i + 1];
                let mut b_grad = NumVector::new(
                    (0..self.dims[i + 1])
                        .map(|j| (0..sample_num).fold(0.0, |acc, k| acc + error_grad[i + 1][k][j]))
                        .collect(),
                );
                if regularization == "l1" {
                    w_grad = &w_grad + &(&self.weight[i].abs_grad() * lambda);
                    b_grad = &b_grad + &(&self.bias[i].abs_grad() * lambda);
                }
                self.weight[i] = &self.weight[i] - &(&w_grad * learning_rate);
                self.bias[i] = &self.bias[i] - &(&b_grad * learning_rate);
            }

            // print progress and error
            if (t + 1) % 100 == 0 {
                println!("{}\t{}", t + 1, obj_func_val);
            }
        }
    }

    // predict and return output layer values
    pub fn predict(&self, x: &NumMatrix) -> NumMatrix {
        assert_eq!(
            x.shape().1,
            self.dims[0],
            "Input data dimension is different"
        );

        // forward prop
        let node_vals = self.forward_prop(&x);
        // return regression value vector
        return node_vals.into_iter().nth(self.layer_num - 1).unwrap();
    }

    // forward propagation
    fn forward_prop(&self, x: &NumMatrix) -> Vec<NumMatrix> {
        let mut node_vals = Vec::new();
        // node value at input layer
        node_vals.push(x.clone());

        // forward prop
        for i in 0..(self.layer_num - 1) {
            // u[i + 1] = v[i] * W[i] + b[i]
            // v[i + 1] = h(u[i + 1])
            let mut result = &node_vals[i] * &self.weight[i];
            for j in 0..x.shape().0 {
                result[j] = &result[j] + &self.bias[i];
            }

            if i < self.layer_num - 2 {
                node_vals.push(result.re_lu());
            } else {
                node_vals.push(result.sigmoid());
            }
        }

        return node_vals;
    }

    // back_propagation
    fn back_prop(&self, values: &NumMatrix, node_vals: &Vec<NumMatrix>) -> Vec<NumMatrix> {
        let mut error_grad = Vec::new();

        // error gradient at output layer
        // objective function = MSE
        error_grad.push(node_vals[self.layer_num - 1].mse_grad(values));

        // back_prop
        for i in (0..(self.layer_num - 1)).rev() {
            // e[i] = (e[i + 1] * W[i].T) . h'(u[i])
            let mut result = &error_grad[self.layer_num - 2 - i] * &self.weight[i].transpose();
            result = result.hadamard_prod(&node_vals[i].re_lu_grad());
            error_grad.push(result);
        }
        error_grad.reverse();

        return error_grad;
    }
}
