# Optional third-party experiment baselines

The `hello_ot` wheel does not contain third-party experiment implementations.

To run the optional HiRef exactness baseline, clone its official repository:

```bash
git clone https://github.com/raphael-group/HiRef third_party/HiRef
```

The adapter uses the upstream torch implementation under `third_party/HiRef/src`.
The upstream repository currently does not expose a license file, so its source
is fetched directly by users rather than redistributed by HELLO.
