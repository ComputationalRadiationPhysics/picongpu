# About

The script generates the Markdown table for compiler compatibility for back-ends for the alpaka `README.md`. It reads the properties from the `supported_compiler.json` and outputs the Markdown table to stdout.

```bash
./generate_supported_compilers.py
```

The generated Markdown can be copied to the alpaka `README.md`.

# Configuration File

The configuration file contains a dictionary. Each key in the dictionary is a compiler. The values contain information about the compatibility with the back-ends. The names of the back-ends are specified by the script. Each back-end requires a `state` property. The `comment` property is optional.
