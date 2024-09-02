#[derive(Debug, Clone)]
pub struct DataOwned<S>
where
    S: Sized,
{
    pub(crate) storage: S,
}

#[derive(Debug, Clone)]
pub struct DataRef<'a, S> {
    pub(crate) storage: &'a S,
}

#[derive(Debug)]
pub struct DataRefMut<'a, S> {
    pub(crate) storage: &'a mut S,
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
    #[inline]
    fn from(data: S) -> Self {
        Self { storage: data }
    }
}

impl<S> DataOwned<S> {
    #[inline]
    pub fn into_storage(self) -> S {
        self.storage
    }
}

pub trait DataAPI {
    type Data: Clone;
    fn storage(&self) -> &Self::Data;
    fn as_ref(&self) -> DataRef<Self::Data>;
    fn into_owned(self) -> DataOwned<Self::Data>;
}

pub trait DataMutAPI: DataAPI {
    fn as_storage_mut(&mut self) -> &mut Self::Data;
    fn as_ref_mut(&mut self) -> DataRefMut<Self::Data>;
}

/* #region impl DataAPI */

impl<S> DataAPI for DataOwned<S>
where
    S: Clone,
{
    type Data = S;
    #[inline]
    fn storage(&self) -> &Self::Data {
        &self.storage
    }
    #[inline]
    fn as_ref(&self) -> DataRef<Self::Data> {
        DataRef { storage: &self.storage }
    }
    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        self
    }
}

impl<'a, S> DataAPI for DataRef<'a, S>
where
    S: Clone,
{
    type Data = S;
    #[inline]
    fn storage(&self) -> &Self::Data {
        self.storage
    }
    #[inline]
    fn as_ref(&self) -> DataRef<Self::Data> {
        self.clone()
    }
    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned { storage: self.storage.clone() }
    }
}

impl<'a, S> DataAPI for DataRefMut<'a, S>
where
    S: Clone,
{
    type Data = S;
    #[inline]
    fn storage(&self) -> &Self::Data {
        self.storage
    }
    #[inline]
    fn as_ref(&self) -> DataRef<'_, Self::Data> {
        DataRef { storage: self.storage }
    }
    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned { storage: self.storage.clone() }
    }
}

impl<'a, S> DataAPI for DataCow<'a, S>
where
    S: Clone,
{
    type Data = S;
    #[inline]
    fn storage(&self) -> &Self::Data {
        match self {
            DataCow::Owned(data) => data.storage(),
            DataCow::Ref(data) => data.storage(),
        }
    }
    #[inline]
    fn as_ref(&self) -> DataRef<Self::Data> {
        match self {
            DataCow::Owned(data) => data.as_ref(),
            DataCow::Ref(data) => data.as_ref(),
        }
    }
    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        match self {
            DataCow::Owned(data) => data,
            DataCow::Ref(data) => data.into_owned(),
        }
    }
}

/* #endregion */

/* #region impl DataMutAPI */

impl<S> DataMutAPI for DataOwned<S>
where
    S: Clone,
{
    #[inline]
    fn as_storage_mut(&mut self) -> &mut Self::Data {
        &mut self.storage
    }
    #[inline]
    fn as_ref_mut(&mut self) -> DataRefMut<Self::Data> {
        DataRefMut { storage: &mut self.storage }
    }
}

impl<'a, S> DataMutAPI for DataRefMut<'a, S>
where
    S: Clone,
{
    #[inline]
    fn as_storage_mut(&mut self) -> &mut Self::Data {
        self.storage
    }
    #[inline]
    fn as_ref_mut(&mut self) -> DataRefMut<'_, Self::Data> {
        DataRefMut { storage: self.storage }
    }
}

/* #endregion */

/* #region DataCow */

impl<S> From<DataOwned<S>> for DataCow<'_, S> {
    #[inline]
    fn from(data: DataOwned<S>) -> Self {
        DataCow::Owned(data)
    }
}

impl<'a, S> From<DataRef<'a, S>> for DataCow<'a, S> {
    #[inline]
    fn from(data: DataRef<'a, S>) -> Self {
        DataCow::Ref(data)
    }
}

/* #endregion */

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
