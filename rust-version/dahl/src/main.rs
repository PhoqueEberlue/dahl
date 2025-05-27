use ndarray::Array3;

mod layers;

fn main() {
    println!("Hello, world!");

    let ouais = Array3::<u32>::zeros((1,2,5));

    println!("{:?}", ouais.shape());
}
