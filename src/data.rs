pub struct DataOwned<S> {
    data: S,
}

pub struct DataRef<'a, S> {
    data: &'a S,
}

pub struct DataRefMut<'a, S> {
    data: &'a mut S,
}

pub enum DataMutable<'a, S> {
    Owned(DataOwned<S>),
    RefMut(DataRefMut<'a, S>),
    ToBeCloned(DataRef<'a, S>, DataOwned<S>),
}

pub enum DataCow<'a, S> {
    Owned(DataOwned<S>),
    Ref(DataRef<'a, S>),
}

pub enum DataReference<'a, S> {
    Ref(DataRef<'a, S>),
    RefMut(DataRefMut<'a, S>),
}

pub trait TraitData {
    type Data;
    fn as_ref(&self) -> DataRef<Self::Data>;
    fn into_owned(self) -> DataOwned<Self::Data>;
}

pub trait TraitDataMut {
    type Data;
    fn as_ref_mut(&mut self) -> DataRefMut<Self::Data>;
}
