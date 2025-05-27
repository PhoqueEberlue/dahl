use ndarray::{Array2, Array3};
use super::Fp;

struct Convolution {
    input_shape: [usize; 2],
    input_data: Array2<Fp>,

    num_filters: usize,
    filter_size: usize,

    filters_shape: [usize; 3],
    output_shape: [usize; 3],

    filters: Array3<Fp>,
    biases: Array3<Fp>,
}

impl Convolution {
    fn new(input_shape: [usize; 2], filter_size: usize, num_filters: usize) -> Self {

        filters_shape = []

        Convolution { input_shape: (), input_data: (), num_filters: (), filter_size: (), filters_shape: (), output_shape: (), filters: (), biases: () }
        
    }
}
