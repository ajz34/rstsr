use crate::prelude_dev::*;

pub trait OpAssignAPI<T, D1, D2>
where
    D1: DimAPI,
    D2: DimAPI,
    Self: DeviceAPI<T>,
{
    fn assign(
        &self,
        a: &mut Storage<T, Self>,
        la: &Layout<D1>,
        b: &Storage<T, Self>,
        lb: &Layout<D2>,
    ) -> Result<()>;
}
