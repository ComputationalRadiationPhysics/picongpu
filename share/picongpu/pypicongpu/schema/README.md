[JSON schemata](https://json-schema.org/) for data passed from PyPIConGPU internal representation to the templating engine

Please refer to PyPIConGPU documentation for more details.

Note that while filenames are *technically* arbitrary (`$id` is used for identification), the file structure follows `lib/python/picongpu/pypicongpu`, with `FILENAME.CLASSNAME.json` for filenames.
(Only one schema/`$id` per json schema file can be handled.)
