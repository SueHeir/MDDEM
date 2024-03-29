use nalgebra::Vector3;

pub struct Gravity {
    gravity_force: Vector3<f64>,
}

impl Gravity {
    pub fn new(values: Vec<&str>) -> Self {
        Gravity {
            gravity_force: Vector3::new(
                values[2].parse::<f64>().unwrap(),
                values[3].parse::<f64>().unwrap(),
                values[4].parse::<f64>().unwrap(),
            ),
        }
    }
}

impl super::Fix for Gravity {
    fn pre_inital_integration(&self) {}

    fn inital_integration(&self) {}

    fn post_inital_integration(&self) {}

    fn pre_exchange(&self) {}

    fn pre_neighbor(&self) {}

    fn pre_force(&self) {}

    fn force(&self) {}

    fn post_force(&self) {}

    fn pre_final_integration(&self) {}

    fn final_integration(&self) {}

    fn post_final_integration(&self) {}
}
