# HTTP Asset Download

## Usage

Run the app. The demo uses libCURL (https://github.com/curl/curl) to download an image from Nabla's **media/color_space_test folder** and a mesh from **media/ply** folders respectively, and then display it on screen.

### Assets
Both assets come from the [Nabla-Example-And-Tests-Media](https://github.com/Devsh-Graphics-Programming/Nabla-Example-And-Tests-Media) repository, pinned to commit `f48f2afe20fee6b356be4c1d922393c751d4dfa7`:

| Asset | URL |
|-------|-----|
| Image (`R8G8B8_1.jpg`) | `https://raw.githubusercontent.com/Devsh-Graphics-Programming/Nabla-Example-And-Tests-Media/f48f2afe20fee6b356be4c1d922393c751d4dfa7/color_space_test/R8G8B8_1.jpg` |
| Mesh (`Spanner-ply.ply`) | `https://github.com/Devsh-Graphics-Programming/Nabla-Example-And-Tests-Media/raw/f48f2afe20fee6b356be4c1d922393c751d4dfa7/ply/Spanner-ply.ply` |

The URLs are the `ImageFileUrl` and `PlyFileUrl` members at the top of `main.cpp`, so you can point the demo at other files by changing them. Both downloads start when the app initializes and run in parallel on the download thread. The window opens right away and shows the clear colour until the image arrives.

### Controls
Press a key to change what is displayed:

| Key | Display |
|-----|---------|
| `1` | The downloaded image (the default view) |
| `2` | The downloaded mesh, spinning on screen |

If you press `2` while the mesh is still downloading, the view switches as soon as it has loaded. If a download fails, an error is logged and that view stays unavailable for the rest of the run, since the demo does not retry.

## API Description:
Usage follows the following pattern:
1. Create an instance of CHttpDownloader (say `m_downloader`) and pass a logger object to it. The constructor starts the download thread, which runs all transfers with libcurl's multi interface.
2. Call `m_downloader.downloadFile(url, callback)` as many times as needed. It never blocks: the request is queued and the call returns immediately. The callback has the signature `void(bool success, std::vector<uint8_t>&& data)`. The callback owns the bytes, so it can `std::move` them on without a copy. On failure `success` is `false` and `data` is empty.
3. Call `m_downloader.update()` from your main thread, typically once per frame. It runs the callbacks of all downloads that have finished since the last call. Callbacks only ever run inside `update()`, on the thread that calls it, so they can safely touch your application state. If you never call `update()`, no callback will fire.
4. Convert the downloaded bytes to an `IFile` with `CHttpDownloader::createFile(path&& filePath, std::vector<uint8_t>&& data)`, passing the bytes with `std::move`. It is a public static function, so you don't need a downloader instance to call it. It takes ownership of `data` and wraps that same buffer in a read-only, in-memory file (a `CFileView`), so nothing is copied; the buffer is freed when the file is destroyed. It returns a `core::smart_refctd_ptr<IFile>`, or `nullptr` if `data` is empty. `filePath` becomes the file's name, and the asset manager uses its extension to pick a loader, so pass something like `path(url).filename()`. Call it inside your download callback, then hand the file to the asset manager, for example `m_assetMgr->getAsset(file.get(), file->getFileName().string(), params)`.

```cpp
CHttpDownloader m_downloader(smart_refctd_ptr(m_logger));

m_downloader.downloadFile(url, [this](bool success, std::vector<uint8_t>&& data)
{
    if (!success)
        return;
    auto file = CHttpDownloader::createFile(path(url).filename(), std::move(data));
    // load `file` with the asset manager, upload to the GPU, ...
});

// every frame:
m_downloader.update();
```