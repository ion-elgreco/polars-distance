Hellooo :)

This plugin is a work-in progress, main goal is to get pairwise distance metrics on numerical vectors (list, arrays) and string distance metrics.

## Examples

```python
import polars
import polars_distance as pld

df = pl.DataFrame({
    "foo":"hello",
    "bar":"hella world"
})

df.select(
    pld.col("foo").pdist_str.hamming('bar').alias('dist')
)
┌──────┐
│ dist │
│ ---  │
│ u32  │
╞══════╡
│ 1    │
└──────┘


df.select(
    pld.col('foo').pdist_str.levenshtein('bar').alias('dist')
)
┌──────┐
│ dist │
│ ---  │
│ u32  │
╞══════╡
│ 6    │
└──────┘

```