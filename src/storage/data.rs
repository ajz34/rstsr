#[derive(Debug, Clone)]
pub struct DataOwned<S>
where
    S: Sized,
{
    data: S,
}

#[derive(Debug, Clone)]
pub struct DataRef<'a, S> {
    data: &'a S,
}

#[derive(Debug)]
pub struct DataRefMut<'a, S> {
    data: &'a mut S,
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
        Self { data }
    }
}

pub trait DataAPI {
    type Data;
    fn as_ref(&self) -> DataRef<Self::Data>;
    fn into_owned(self) -> DataOwned<Self::Data>;
}

pub trait DataMutAPI {
    type Data;
    fn as_ref_mut(&mut self) -> DataRefMut<Self::Data>;
}

impl<S> DataAPI for DataOwned<S> {
    type Data = S;
    fn as_ref(&self) -> DataRef<Self::Data> {
        DataRef { data: &self.data }
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
    fn as_ref(&self) -> DataRef<Self::Data> {
        self.clone()
    }
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned { data: self.data.clone() }
    }
}

impl<'a, S> DataAPI for DataRefMut<'a, S>
where
    S: Clone,
{
    type Data = S;
    fn as_ref(&self) -> DataRef<'_, Self::Data> {
        DataRef { data: &self.data }
    }
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned { data: self.data.clone() }
    }
}

impl<S> DataMutAPI for DataOwned<S> {
    type Data = S;
    fn as_ref_mut(&mut self) -> DataRefMut<Self::Data> {
        DataRefMut { data: &mut self.data }
    }
}

impl<'a, S> DataMutAPI for DataRefMut<'a, S> {
    type Data = S;
    fn as_ref_mut(&mut self) -> DataRefMut<'_, Self::Data> {
        DataRefMut { data: &mut self.data }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_trait_data() {
        let vec = vec![10, 20, 30];
        println!("{:?}", vec.as_ptr());
        let data = DataOwned { data: vec };
        let data_ref = data.as_ref();
        let data_ref_ref = data_ref.as_ref();
        println!("{:?}", data_ref.data.as_ptr());
        println!("{:?}", data_ref_ref.data.as_ptr());
        // let data2 = data.into_owned();
        let data_ref2 = data_ref.into_owned();
        // println!("{:?}", data2.data.as_ptr());
        println!("{:?}", data_ref2.data.as_ptr());
    }
}
