# Color Space Test

## Usage

Just launch the executable to display images from input list. Application supports following command-line arguments:

- `--verbose`: Enables detailed logging. Useful for debugging or understanding the app's process.
- `--test`: Performs tests by comparing the current data with reference data saved on disk. 
- `--update-references`: Updates reference files with the current data. 
- `--input-list`: Overrides default path to input list with images to run the application with.

## CTest

### JSON output format

The application can be tested with `hash` mode which outputs JSON files containing image hash values for image asset. Here's an example of a JSON output.

```json
{
    "image": [
        18371015033211807487,
        861795556807058655,
        16487203311713844624,
        15144584759684060250
    ]
}
```

### Fire tests!

Build the application with desired configuration, `cd` to it's build directory and fire

```bash
ctest -C <configuration> --progress --stop-on-failure --verbose
```

test writes current image hash to a file in *cwd* and compares with saved reference file - if there is a difference the test fails.
