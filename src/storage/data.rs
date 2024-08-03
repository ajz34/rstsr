#[derive(Debug, Clone)]
pub struct DataOwned<S>
where
    S: Sized,
{
    storage: S,
}

#[derive(Debug, Clone)]
pub struct DataRef<'a, S> {
    storage: &'a S,
}

#[derive(Debug)]
pub struct DataRefMut<'a, S> {
    storage: &'a mut S,
}

#[derive(Debug)]
pub enum DataMutable<'a, S> {
    Owned(DataOwned<S>),
    RefMut(DataRefMut<'a, S>),
    ToBeCloned(DataRef<'a, S>, DataOwned<S>),
}

#[derive(Debug, Clone)]
pub enum DataCow<'a, S> {
    Owned(DataOwned<S>),
    Ref(DataRef<'a, S>),
}

#[derive(Debug)]
pub enum DataReference<'a, S> {
    Ref(DataRef<'a, S>),
    RefMut(DataRefMut<'a, S>),
}

impl<S> From<S> for DataOwned<S> {
    fn from(data: S) -> Self {
        Self { storage: data }
    }
}

impl<S> DataOwned<S> {
    pub fn into_storage(self) -> S {
        self.storage
    }
}

pub trait DataAPI {
    type Data;
    fn as_storage(&self) -> &Self::Data;
    fn as_ref(&self) -> DataRef<Self::Data>;
    fn into_owned(self) -> DataOwned<Self::Data>;
}

pub trait DataMutAPI {
    type Data;
    fn as_ref_mut(&mut self) -> DataRefMut<Self::Data>;
}

impl<S> DataAPI for DataOwned<S> {
    type Data = S;
    fn as_storage(&self) -> &Self::Data {
        &self.storage
    }
    fn as_ref(&self) -> DataRef<Self::Data> {
        DataRef { storage: &self.storage }
    }
    fn into_owned(self) -> DataOwned<Self::Data> {
        self
    }
}

impl<'a, S> DataAPI for DataRef<'a, S>
where
    S: Clone,
{
    type Data = S;
    fn as_storage(&self) -> &Self::Data {
        self.storage
    }
    fn as_ref(&self) -> DataRef<Self::Data> {
        self.clone()
    }
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned { storage: self.storage.clone() }
    }
}

impl<'a, S> DataAPI for DataRefMut<'a, S>
where
    S: Clone,
{
    type Data = S;
    fn as_storage(&self) -> &Self::Data {
        self.storage
    }
    fn as_ref(&self) -> DataRef<'_, Self::Data> {
        DataRef { storage: &self.storage }
    }
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned { storage: self.storage.clone() }
    }
}

impl<S> DataMutAPI for DataOwned<S> {
    type Data = S;
    fn as_ref_mut(&mut self) -> DataRefMut<Self::Data> {
        DataRefMut { storage: &mut self.storage }
    }
}

impl<'a, S> DataMutAPI for DataRefMut<'a, S> {
    type Data = S;
    fn as_ref_mut(&mut self) -> DataRefMut<'_, Self::Data> {
        DataRefMut { storage: &mut self.storage }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_trait_data() {
        let vec = vec![10, 20, 30];
        println!("{:?}", vec.as_ptr());
        let data = DataOwned { storage: vec };
        let data_ref = data.as_ref();
        let data_ref_ref = data_ref.as_ref();
        println!("{:?}", data_ref.storage.as_ptr());
        println!("{:?}", data_ref_ref.storage.as_ptr());
        // let data2 = data.into_owned();
        let data_ref2 = data_ref.into_owned();
        // println!("{:?}", data2.data.as_ptr());
        println!("{:?}", data_ref2.storage.as_ptr());
    }
}
