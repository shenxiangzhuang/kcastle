# Framework migration / 框架迁移

中文与 English 混排，**粗体关键词**、*斜体*、~~删除线~~ 和 `inline_code`。Inline formula $x^2 + y^2 = z^2$ ends here. Emoji 😀 中文公式 $\text{中文} + \text{😀}$。

## Lists and quote

98. First item with $x_i$.
99. Second item with **bold text** and wrapping content that should remain aligned correctly on a narrow window.
100. Third item.

- [x] Completed
- [ ] Pending

> Quoted text with $\frac{a}{b}$ and **bold**.
> A second line in the same paragraph.

## Formula table

| Operation | Formula | Explanation |
| --- | --- | --- |
| Attention | $O(t^2 d)$ | 中文说明 English explanation |
| Total | $\sum_{t=1}^{T} O(t^2) = O(T^3)$ | A long description that wraps |

$$
\int_0^1 x^2 dx = \frac{1}{3}
$$

## Code and copy

```haskell
quicksort [] = []
quicksort (x:xs) = quicksort [a | a <- xs, a < x] ++ [x] ++ quicksort [a | a <- xs, a >= x]
```

LaTeX inline \(a+b\), display \[E=mc^2\], literal `\(code\)`.

---

FINAL SENTINEL 最后一段，不应丢失。
