use std::fmt;

use iree_sys as sys;

/// HAL element type (`iree_hal_element_type_t`): numerical type + bit width.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElementType(sys::iree_hal_element_type_t);

impl ElementType {
    pub const NONE: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_NONE);
    pub const BOOL8: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_BOOL_8);
    /// Signless 8-bit integer (`i8` in MLIR).
    pub const INT8: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_INT_8);
    pub const SINT8: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_SINT_8);
    pub const UINT8: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_UINT_8);
    pub const INT16: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_INT_16);
    pub const SINT16: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_SINT_16);
    pub const UINT16: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_UINT_16);
    pub const INT32: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_INT_32);
    pub const SINT32: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_SINT_32);
    pub const UINT32: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_UINT_32);
    pub const INT64: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_INT_64);
    pub const SINT64: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_SINT_64);
    pub const UINT64: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_UINT_64);
    pub const FLOAT16: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_FLOAT_16);
    pub const BFLOAT16: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_BFLOAT_16);
    pub const FLOAT32: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_FLOAT_32);
    pub const FLOAT64: Self = Self(sys::IREE_HAL_ELEMENT_TYPE_FLOAT_64);

    pub const fn from_raw(raw: sys::iree_hal_element_type_t) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> sys::iree_hal_element_type_t {
        self.0
    }

    pub const fn bit_count(self) -> u32 {
        sys::iree_hal_element_bit_count(self.0)
    }

    /// Byte size of one element, or `None` for sub-byte / non-byte-aligned types
    /// (e.g. `i4`), which this crate does not marshal.
    pub const fn byte_size(self) -> Option<usize> {
        let bits = self.bit_count();
        if bits != 0 && bits.is_multiple_of(8) {
            Some((bits / 8) as usize)
        } else {
            None
        }
    }

    /// True for signless / signed / unsigned integers and booleans.
    pub const fn is_integer(self) -> bool {
        (sys::iree_hal_element_numerical_type(self.0) & 0xF0)
            == sys::IREE_HAL_NUMERICAL_TYPE_INTEGER
    }

    /// Mirrors the runtime's ABI check (`iree_hal_element_types_are_compatible`):
    /// integers of the same width are interchangeable (signless vs signed vs
    /// unsigned); everything else must match exactly.
    pub const fn is_compatible_with(self, other: Self) -> bool {
        if self.is_integer() && other.is_integer() {
            self.bit_count() == other.bit_count()
        } else {
            self.0 == other.0
        }
    }

    fn name(self) -> Option<&'static str> {
        Some(match self {
            Self::NONE => "none",
            Self::BOOL8 => "i1(bool8)",
            Self::INT8 => "i8",
            Self::SINT8 => "si8",
            Self::UINT8 => "ui8",
            Self::INT16 => "i16",
            Self::SINT16 => "si16",
            Self::UINT16 => "ui16",
            Self::INT32 => "i32",
            Self::SINT32 => "si32",
            Self::UINT32 => "ui32",
            Self::INT64 => "i64",
            Self::SINT64 => "si64",
            Self::UINT64 => "ui64",
            Self::FLOAT16 => "f16",
            Self::BFLOAT16 => "bf16",
            Self::FLOAT32 => "f32",
            Self::FLOAT64 => "f64",
            _ => return None,
        })
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => f.write_str(name),
            None => write!(f, "element_type(0x{:08X})", self.0),
        }
    }
}

impl fmt::Debug for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A Rust scalar type with a fixed HAL element type that can be copied to and
/// from device buffers bit-for-bit.
///
/// # Safety
///
/// The trait is sealed; for the (in-crate) implementors:
/// `Self` must be plain-old-data: every bit pattern is valid, no padding, and
/// `size_of::<Self>() * 8 == ELEMENT_TYPE.bit_count()`.
pub unsafe trait Element: sealed::Sealed + Copy + Send + Sync + 'static {
    const ELEMENT_TYPE: ElementType;
}

macro_rules! element {
    ($($t:ty => $e:ident),* $(,)?) => {$(
        impl sealed::Sealed for $t {}
        // SAFETY: primitive POD type whose width matches the element type.
        unsafe impl Element for $t {
            const ELEMENT_TYPE: ElementType = ElementType::$e;
        }
        const _: () = assert!(std::mem::size_of::<$t>() * 8 == ElementType::$e.bit_count() as usize);
    )*};
}

element!(
    i8 => SINT8,
    u8 => UINT8,
    i16 => SINT16,
    u16 => UINT16,
    i32 => SINT32,
    u32 => UINT32,
    i64 => SINT64,
    u64 => UINT64,
    f32 => FLOAT32,
    f64 => FLOAT64,
);

#[cfg(feature = "half")]
element!(half::f16 => FLOAT16, half::bf16 => BFLOAT16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_compatibility_mirrors_runtime() {
        assert!(ElementType::INT64.is_compatible_with(ElementType::SINT64));
        assert!(ElementType::UINT32.is_compatible_with(ElementType::INT32));
        assert!(!ElementType::INT32.is_compatible_with(ElementType::INT64));
        assert!(!ElementType::FLOAT32.is_compatible_with(ElementType::INT32));
        assert!(!ElementType::FLOAT16.is_compatible_with(ElementType::BFLOAT16));
        assert!(ElementType::FLOAT32.is_compatible_with(ElementType::FLOAT32));
    }

    #[test]
    fn sizes() {
        assert_eq!(ElementType::FLOAT16.byte_size(), Some(2));
        assert_eq!(ElementType::INT64.byte_size(), Some(8));
        assert_eq!(ElementType::BOOL8.byte_size(), Some(1));
        assert_eq!(
            ElementType::from_raw(sys::iree_hal_make_element_type(0x10, 4)).byte_size(),
            None
        );
        assert_eq!(ElementType::FLOAT32.to_string(), "f32");
    }
}
