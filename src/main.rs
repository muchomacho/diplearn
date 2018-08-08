/*
    read breast X-ray picture data and predict whether malignant tumor exists in ROI(region of interest)
*/

#[allow(unused_imports)]
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
#[allow(unused_imports)]
use std::env;
#[allow(unused_imports)]
use std::fs::File;
#[allow(unused_imports)]
use std::io::prelude::*;
#[allow(unused_imports)]
use std::io::{stdin, stdout, BufReader, BufWriter, Write};

extern crate rand;
use rand::{seq, thread_rng};

extern crate diplearn;
#[allow(unused_imports)]
use diplearn::mlp::MLP;
#[allow(unused_imports)]
use diplearn::numrs::*;

// buffer size for file I/O
static BUF_SIZE: usize = 100 * 1024;
static LINE_SIZE: usize = 1024;

// dimension of each layer
static LAYER_DIMS: [usize; 5] = [117, 8, 8, 4, 1];
// learning epoch
static EPOCH_NUM: usize = 20000;
// learning rate
static LEARNING_RATE: f64 = 0.05;
// coef of regularization term
static REG_COEF: f64 = 0.0000000001;
// sampling bias
static SAMPLE_BIAS: f64 = 4.0;
// bias for deciding answer
static PROB_COEF: f64 = 1.0;
// number of kinds of probability bias
static PROB_NUM: usize = 300;

fn main() {
    // ---------------------------------------------------------------------------
    // read values of data
    let f = File::open("training_data/Info.txt").unwrap();
    let mut reader = BufReader::with_capacity(BUF_SIZE, f);
    let mut values = Vec::new();
    let mut line = String::with_capacity(LINE_SIZE);
    while reader.read_line(&mut line).unwrap() > 0 {
        {
            let raw_value = line.trim()
                .split_whitespace()
                .map(|s| s.trim().parse::<f64>().unwrap())
                .nth(0)
                .unwrap();
            if raw_value > 0.0 {
                values.push(vec![1.0]);
            } else {
                values.push(vec![0.0]);
            }
        }
        line.clear();
    }

    // ---------------------------------------------------------------------------
    // read feature vectors of data
    let f = File::open("training_data/Features.txt").unwrap();
    let mut reader = BufReader::with_capacity(BUF_SIZE, f);
    let mut vectors: Vec<Vec<f64>> = Vec::new();
    while reader.read_line(&mut line).unwrap() > 0 {
        {
            let vec = line.trim()
                .split_whitespace()
                .map(|s| s.trim().parse::<f64>().unwrap())
                .collect();
            vectors.push(vec);
        }
        line.clear();
    }

    // ---------------------------------------------------------------------------
    // devide data into training and test randomly
    let mut train_vectors = Vec::new();
    let mut train_values = Vec::new();
    let mut test_vectors = Vec::new();
    let mut test_values = Vec::new();

    // shuffle indices
    // half of data is used for training, the rest half is used for test
    let mut rng = thread_rng();
    let random_index = seq::sample_indices(&mut rng, vectors.len(), vectors.len());

    // decrease negative data because dataset has much more negative data than positive ones
    // select training data so that #positive : #negative = 1 : SAMPLE_BIAS
    let mut positive = 0;
    for i in 0..(vectors.len() / 2) {
        if values[random_index[i]][0] == 1.0 {
            positive += 1;
        }
    }
    println!(
        "training data\npositive sample = {}\nnegative sample = {}\n",
        positive,
        vectors.len() / 2 - positive
    );

    let mut negative_count = 0;
    for i in 0..(vectors.len() / 2) {
        if values[random_index[i]][0] == 1.0 {
            train_vectors.push(vectors[random_index[i]].clone());
            train_values.push(values[random_index[i]].clone());
        } else {
            if negative_count <= (positive as f64 * SAMPLE_BIAS) as usize {
                train_vectors.push(vectors[random_index[i]].clone());
                train_values.push(values[random_index[i]].clone());
                negative_count += 1;
            }
        }
    }

    // create test data
    for i in (vectors.len() / 2)..vectors.len() {
        test_vectors.push(vectors[random_index[i]].clone());
        test_values.push(values[random_index[i]].clone());
    }
    let train_vectors = NumMatrix::new(train_vectors);
    let train_values = NumMatrix::new(train_values);
    let test_vectors = NumMatrix::new(test_vectors);
    let test_values = NumMatrix::new(test_values);

    // ---------------------------------------------------------------------------
    // train
    let mut learner = MLP::new(LAYER_DIMS.to_vec());
    learner.train(
        &train_vectors,
        &train_values,
        EPOCH_NUM,
        LEARNING_RATE,
        "l1",
        REG_COEF,
    );

    // ---------------------------------------------------------------------------
    // predict
    let test_sample_size = test_values.shape().0;
    let out_prob = learner.predict(&test_vectors);
    let mut answer = vec![Vec::new(); PROB_NUM];
    // decide answer with bias
    // prob_bias * prob > 0.5 -> positive
    for i in 0..test_sample_size {
        for j in 0..PROB_NUM {
            let prob_bias = PROB_COEF * (j + 1) as f64;
            if prob_bias * out_prob[i][0] > 0.5 {
                answer[j].push(1.0);
            } else {
                answer[j].push(0.0);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // check whether the answers are correct
    let mut correct_answer = vec![0; PROB_NUM];
    let mut true_positive = vec![0; PROB_NUM];
    let mut positive = 0;
    let mut negative = 0;
    let mut positive_mean = 0.0;
    let mut negative_mean = 0.0;
    for i in 0..test_sample_size {
        if test_values[i][0] == 1.0 {
            positive += 1;
            positive_mean += out_prob[i][0];
        } else {
            negative += 1;
            negative_mean += out_prob[i][0];
        }
        for j in 0..PROB_NUM {
            if answer[j][i] == test_values[i][0] {
                correct_answer[j] += 1;
                if test_values[i][0] == 1.0 {
                    true_positive[j] += 1;
                }
            }
        }
    }

    println!(
        "\nPositive mean = {}\nNegative mean = {}\n",
        positive_mean / positive as f64,
        negative_mean / negative as f64
    );
    println!("Probability_bias\tPrecision\tSensitivity");
    for i in 0..PROB_NUM {
        println!(
            "{}\t{}\t{}",
            PROB_COEF * (i + 1) as f64,
            correct_answer[i] as f64 / test_sample_size as f64,
            true_positive[i] as f64 / positive as f64
        );
    }
}
