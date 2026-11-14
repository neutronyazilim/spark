use std::{mem, slice};

/// A fixed-capacity bump allocator for inference scratch memory.
///
/// Allocation is monotonic until [`Arena::reset`] is called. It never grows
/// and therefore does not allocate during token generation.
#[allow(dead_code)]
pub struct Arena {
    storage: Vec<u8>,
    cursor: usize,
}

#[allow(dead_code)]
impl Arena {
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: vec![0; capacity],
            cursor: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.storage.len()
    }

    pub fn used(&self) -> usize {
        self.cursor
    }

    pub fn remaining(&self) -> usize {
        self.capacity() - self.cursor
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn alloc_bytes(&mut self, len: usize, align: usize) -> Option<&mut [u8]> {
        let start = self.aligned_start(len, align)?;
        self.cursor = start + len;
        Some(&mut self.storage[start..self.cursor])
    }

    pub fn alloc_slice<T: Copy + Default>(&mut self, len: usize) -> Option<&mut [T]> {
        let size = mem::size_of::<T>().checked_mul(len)?;
        let start = self.aligned_start(size, mem::align_of::<T>())?;
        self.cursor = start + size;

        // The arena storage is initialized, and every element is explicitly
        // initialized before the typed slice is exposed.
        let ptr = self.storage.as_mut_ptr().wrapping_add(start).cast::<T>();
        // SAFETY: `start` is aligned for T, the range is inside storage, and
        // the range does not overlap any previously returned allocation.
        let result = unsafe { slice::from_raw_parts_mut(ptr, len) };
        result.fill(T::default());
        Some(result)
    }

    fn aligned_start(&self, len: usize, align: usize) -> Option<usize> {
        if !align.is_power_of_two() {
            return None;
        }
        let base = self.storage.as_ptr() as usize;
        let current = base.checked_add(self.cursor)?;
        let aligned = (current + (align - 1)) & !(align - 1);
        let start = aligned.checked_sub(base)?;
        let end = start.checked_add(len)?;
        (end <= self.storage.len()).then_some(start)
    }
}

#[cfg(test)]
mod tests {
    use super::Arena;

    #[test]
    fn allocations_are_aligned_and_resettable() {
        let mut arena = Arena::new(128);
        let values = arena.alloc_slice::<f32>(3).expect("f32 allocation");
        values[0] = 1.0;
        assert_eq!(values, &[1.0, 0.0, 0.0]);
        assert!(arena.used() >= 3 * std::mem::size_of::<f32>());
        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn capacity_is_enforced() {
        let mut arena = Arena::new(4);
        assert!(arena.alloc_bytes(4, 1).is_some());
        assert!(arena.alloc_bytes(1, 1).is_none());
    }
}
