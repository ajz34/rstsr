use crate::prelude_dev::*;

pub trait DeviceOpBinary<T, D>
where
    Self: DeviceAPI<T>,
    D: DimAPI,
{
    fn assign(
        &self,
        a: &mut Storage<T, Self>,
        la: &Layout<D>,
        b: &Storage<T, Self>,
        lb: &Layout<D>,
    ) -> Result<()>;
}
