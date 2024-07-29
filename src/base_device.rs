pub trait TraitDevice: Clone + PartialEq {
    fn device(&self) -> Self;
}

pub trait TraitStorage: Clone {
    type Device: TraitDevice;
    type DType;
}
