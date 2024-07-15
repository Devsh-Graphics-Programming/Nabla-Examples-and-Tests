# StreamHashApp

## Description

`StreamHashApp` is a command-line application developed as part of the Nabla Engine. It's designed to generate hash values for image assets using various execution policies. The application supports testing against reference data and updating these references as needed. Currently the input image is hardcoded but we may extend it to reference any input image format supported by the engine.

## JSON output format

The application outputs JSON files containing hash values for image asset. Here's an example of the JSON output format.

```json
{
    "image": [
        18371015033211807487,
        861795556807058655,
        16487203311713844624,
        15144584759684060250
    ],
    "mipLevels": [
        {
            "layers": [
                {
                    "hash": [
                        14913208377704045379,
                        17430601461277626947,
                        12187504251803418887,
                        5282962499579990907
                    ]
                },
                
                <...>
                
                {
                    "hash": [
                        18345509054482818774,
                        16330985058033662834,
                        12288694599209351308,
                        15537853624489125915
                    ]
                }
            ]
        },
        
        <...>
        
        {
            "layers": [
                {
                    "hash": [
                        14913208377704045379,
                        17430601461277626947,
                        12187504251803418887,
                        5282962499579990907
                    ]
                },
                
                <...>
                
                {
                    "hash": [
                        18345509054482818774,
                        16330985058033662834,
                        12288694599209351308,
                        15537853624489125915
                    ]
                }
            ]
        }
    ],
    "policy": "std::execution::sequenced_policy"
}
```

This JSON structure includes the overall image hash values, the execution policy used, and detailed hash values for each mip level and layer within the image. Final image hash is a hash of all level & mip level hashes.

## Usage

To use `StreamHashApp`, you can specify the following command-line arguments:

- `--verbose`: Enables detailed logging. Useful for debugging or understanding the app's process.
- `--test`: Performs tests by comparing the current data with reference data. 
- `--update-references`: Updates the JSON reference files with the current data. 

Note CWD must be one level bellow from executable (`.\bin\..`).

## CTest

Build the application with desired configuration and fire

```bash
ctest -C <configuration> --progress --stop-on-failure --verbose
```

currently we have issues with `std::execution::parallel_policy`, turns out the filter hashing image does OOB writes somewhere and crashes the application.