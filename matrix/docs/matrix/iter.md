行列のイテレータを返します。

# Examples

```rust
use matrix::Matrix;
let mtx = Matrix::<i32>::new([[1, 2], [3, 4]]);
let mut iter = mtx.iter();
assert_eq!(iter.next(), Some(&1));
assert_eq!(iter.next(), Some(&2));
assert_eq!(iter.next(), Some(&3));
assert_eq!(iter.next(), Some(&4));
```