// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// always include nabla first before std:: headers
#include "nabla.h"

#include "nbl/system/CStdoutLogger.h"
#include "nbl/system/CFileLogger.h"
#include "nbl/system/CColoredStdoutLoggerWin32.h"
#include "nbl/system/IApplicationFramework.h"

#include <iostream>
#include <cstdio>

// if this config cmake flag is available then we embedded resources using cmake into C++ source
#ifdef NBL_EMBED_BUILTIN_RESOURCES
// these are the engine builtins in `nbl/builtin`
#include "nbl/builtin/CArchive.h"
// this is the one we've made in the example's cmake
#include "yourNamespace/builtin/CArchive.h"
#endif

// In general we will only write out the full namespace for the first occurence of an identifier
using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;

int main(int argc, char** argv)
{
	// the application only needs to call this to delay-load Shared Libraries, if you have a static build, it will do nothing
	nbl::system::IApplicationFramework::GlobalsInit();
	// we will actually use `IApplicationFramework` in later samples

	// A `nbl:::core::smart_refctd_ptr<T>` is our version of `std::shared_ptr<T>` except we require that T inherits from `nbl::core::IReferenceCounted`
	// `IReferenceCounted` is our version of `std::enable_shared_from_this<CRTP>` but it has additional constraints which make the class non-copyable and a pure interface
	// `nbl::system::ISystem` is a Operating System interface, it implements a virtual filesystem on in addition to the real OS one but also abstracts some OS-specific things
	// Nothing in Nabla is a singleton or similar silly "conceled global variable" software design pattern, you can create multiple `ISystem` if you like.
	smart_refctd_ptr<ISystem> system = IApplicationFramework::createSystem();

	// Nabla's virtual filesystem has no notion of a Current Working Directory as its inherently thread-unsafe, everything operates on "absolute" paths
	const nbl::system::path CWD = path(argv[0]).parent_path().generic_string() + "/";

	// we assume you'll run the example `../..` relative to our media dir
	path mediaWD = CWD.generic_string() + "../../media/";
	// but allow an override
	if (argc>=2 && nbl::core::string("-media_dir")==argv[1])
	{
		mediaWD = path(argv[2]);
		if (mediaWD.is_absolute())
			mediaWD = CWD/mediaWD;
	}

	// TODO: system->deleteFile("log.txt");

	// Most APIs in Nabla take a custom logger, this way you can filter your logs from different subsystems or objects.
	smart_refctd_ptr<nbl::system::ILogger> logger;
	// By default loges write out WARNING, PERFORMANCE and ERROR messages, here we make them write out everything
	// The bitflag class exists to make the bit operations on scoped enums a little nicer
	const auto LogLevel = nbl::core::bitflag(ILogger::ELL_DEBUG) | ILogger::ELL_INFO | ILogger::ELL_WARNING | ILogger::ELL_PERFORMANCE | ILogger::ELL_ERROR;
	// In this sample we will write some of the logging output to a file.
	{
		// All File I/O operations in Nabla are deferred onto a dedicated I/O thread.
		// This means all file operations return `ISystem::future_t<ReturnValue>`, which are non-copyable and non-movable, and CANCELLABLE (more on that later).
		// The reason for this is because the operations are queued up on a lockless (unless overflown) circular buffer and need the future's address to remain constant.
		// If you want to pass futures around, then `new` them on the heap, and ideally wrap them in a `unique_ptr`.
		ISystem::future_t<smart_refctd_ptr<nbl::system::IFile>> future;
		// `createFile` creates an `IFile`, whether a new file is created in the filesystem depends on whether it exists and write access permissions
		// the last parameter is a flag indicating what sort of access we intend to perform on the file, this is rigorously checked.
		system->createFile(future, CWD/"log.txt", IFile::ECF_READ_WRITE);
		// A future needs to be awaited before use, `wait()` only returns when the result is `ready()`.
		// THIS DOES NOT MEAN THAT THE OPERATION WAS A SUCCESS, ONLY THAT THE REQUEST WAS PROCESSED.
		// IMPORTANT NOTE: Since `ISystem::future_t` are cancellable, it means that if you don't `wait()` on a future, `~future_t` will cancel the enqueued operation.
		// So if you don't `wait()` or similar, your operation may or may not be performed depending on the delay between enqueue and `~future_t`.
		if (future.wait() && future.get())
		{
			// The `copy()` and `get()` methods are inherently unsafe requiring that access is externally synchronized and `ready()` was true before calling them.
			// You'd usually want to use `acquire()` and move the return value out of the locked storage for 100% safety and peace of mind.
			logger = nbl::core::make_smart_refctd_ptr<system::CFileLogger>(future.copy(),/*append to existing file*/false,LogLevel);
		}
		else
		{
			logger = make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>(LogLevel);
			logger->log("Could not create \"%s\\log.txt\" logging to STDOUT instead!\n",ILogger::ELL_ERROR,CWD.string().c_str());
		}
	}

	// Now onto some tests
	ISystem::future_t<smart_refctd_ptr<IFile>> future;
	system->createFile(future, CWD/"testFile.txt", bitflag(IFile::ECF_READ_WRITE)/*Growing mappable files are a TODO |IFile::ECF_MAPPABLE*/);
	// This time we showcase the "correct" API usage, this is safe against a different thread/agent cancelling your request/future
	// The difference between `acquire()` and `try_acquire()` is that `acquire()` will block until either the request is ready OR fails (gets cancelled, aborted, etc.)
	if (auto pFile = future.acquire(); pFile && pFile->get())
	{
		// NOTE: You can only hold onto the dereferenced storage of a future as long as it is locked 
		auto& file = *pFile;
		const string fileData = "Test file data!";

		// `nbl::system::IFile::success_t` is a utility to cut down on the verbosity of the handling of the `future_t` returned by `IFile::read` and `IFile::write`
		IFile::success_t writeSuccess;
		file->write(writeSuccess, fileData.data(), 0, fileData.length());
		// However just as with the `future_t` the `success_t` needs to be actually awaited or the operation may be cancelled and not be performed.
		// The explicit boolean conversion operators invoke `success_t::getBytesProcessed(block=true)`
		if (!bool(writeSuccess))
			logger->log("Failed to write file %p !",ILogger::ELL_ERROR,file.get());

		string readStr(fileData.length(),'\0');
		IFile::success_t readSuccess;
		file->read(readSuccess, readStr.data(), 0, readStr.length());
		if (!bool(readSuccess))
			logger->log("Failed to read file %p !",ILogger::ELL_ERROR,file.get());

		if (readStr!=fileData)
			logger->log("File %p readback results don't match!",ILogger::ELL_ERROR,file.get());
	}
	else
		logger->log("File \"testFile.txt\" could not be created in CWD!",ILogger::ELL_ERROR);

	// simple lambda to spit out file contents to the log
	auto testFile = [&](const smart_refctd_ptr<IFile>* const pFile) -> bool
	{
		if (pFile)
		{
			// word of warning, unless your file is `const IFile` then the overload of `getMappedPointer()` which will be matched is the non-const one that returns `void*`
			const IFile* file = pFile->get();
			// The `void* getMappedPointer()` checks for the write-access flag which we omitted and would return nullptr to you, so this is why its important `file` be `const IFile*`
			if (file && file->getMappedPointer())
			{
				// the only reason why we even copy the data is because of needing to ensure the null char is present for the logger
				string readStr(file->getSize(), '\0');
				memcpy(readStr.data(), file->getMappedPointer(), file->getSize());
				logger->log("%s\n\n\n\n\n===================================================================\n\n\n\n\n", ILogger::ELL_INFO, readStr.c_str());
				return true;
			}
		}
		return false;
	};
	// simple lambda to spit out file contents to the log
	auto testPath = [&](const string& path) -> void
	{
		ISystem::future_t<smart_refctd_ptr<IFile>> future;
		// Here comes a novelty, memory mapped files. Most IFiles in IArchives get decompressed to memory already so no point in making extra copies.
		// Also as long as you want read-only or write-non-growable access to a file, we can open regular OS-files on disk as memory mapped too!
		system->createFile(future, path.c_str(), bitflag(IFileBase::ECF_READ) | IFileBase::ECF_MAPPABLE);
		if (auto pFile = future.acquire(); !pFile || !testFile(pFile.operator->()))
			logger->log("Supposed built-in \"%s\" could not be opened!", ILogger::ELL_ERROR, path.c_str());
	};
	// Non-development builds on Nabla will embed most files under "include/nbl/builtin" using our CMake utilities into the Library's source code,
	// but whenever `NBL_EMBED_BUILTIN_RESOURCES` is disabled, we'll just mount directories of the SDK under same paths (kinda like a folder symlink).
	{
		testPath("nbl/builtin/glsl/utils/acceleration_structures.glsl"); // nbl internal BRs
		testPath("spirv/unified1/spirv.hpp"); // dxc internal BRs
		testPath("boost/preprocessor.hpp"); // boost preprocessor internal BRs
	}
	// Those same CMake utilities actually allow you to create your own Static and Dynamic libraries of read-only `CArchive` in a custom namespace.
#ifdef NBL_EMBED_BUILTIN_RESOURCES
	{
		// It's not actually recommended practice to put embedded resources in their own standalone DLLs, this is just for testing
		#ifdef _NBL_SHARED_BUILD_
		{
			const auto brOutputDLLAbsoluteDirectory = std::filesystem::absolute(std::filesystem::path(_BR_DLL_DIRECTORY_)).string();
			const HRESULT brLoad = nbl::system::CSystemWin32::delayLoadDLL(_BR_DLL_NAME_, { brOutputDLLAbsoluteDirectory.c_str(), "" });

			assert(SUCCEEDED(brLoad));
		};
		#endif

		// the custom archives need to be manually created by you
		auto archive = make_smart_refctd_ptr<yourNamespace::builtin::CArchive>(smart_refctd_ptr(logger));
		// and mounted. Btw the `mount()` lets you mount the archive under a different alias path.
		system->mount(smart_refctd_ptr(archive));

		// archive path test via ISystem
		testPath("dir/data/test.txt");

		// archive alias and manual retrieval test
		{
			// Archives are funny, because `getFile` is not meant to defer to a worker thread (it might be invoked on a worker thread as a result of `ISystem::createFile` later on).
			// This function will basically perform all necessary decompression or copying (for a memory mapped file) and basically block the caller until done, so there is no `ISystem::future_t` here.
			// CONTRIBUTOR NOTE: We may implement the files in the far future in such a way that they decompress/fill lazily when they page fault.
			smart_refctd_ptr<IFile> file = archive->getFile("aliasTest1",""); // alias to dir/data/test.txt
			
			if (!testFile(&file))
				logger->log("Supposed built-in \"aliasTest1\" could not be opened!", ILogger::ELL_ERROR);
		}
	}
#endif

	// The Asset Manager is a class with which you register loaders for various types of assets (images, buffers, shaders, models, etc.), it also coordinates the different loaders when they depend on each other.
	// For example a Mesh OBJ loader will usually invoke a Graphics Pipeline Loader which reads from MTL files, which will invoke an image loader (PNG, JPG, DDS, KTX). 
	// The raw-pointer constructor of `smart_refctd_ptr` is `explicit`, this is to make the user painfully aware of ownership being taken at a particular moment.
	// To further drive the point home, there is a pattern in Nabla of asking for r-value references to shared pointers in certain functions, instead of l-values or raw-pointers.
	auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(system));

	// when assets are retrieved you need to provide parameters that control the loading process
	nbl::asset::IAssetLoader::SAssetLoadParams lp;
	// at the very least you need to provide the `workingDirectory` if your asset depends on others, this helps resolve relative paths for things such as textures
	lp.workingDirectory = mediaWD;
	// Its good practice to provide a logger as well. Note that there are some raw pointer parameters and you need to make sure their object's lifetimes last until any method using the parameters returns.
	lp.logger = logger.get();
	// Handle failures instead of nasty asserts
	// Nabla's Asset objects form a Directed Acyclic Graph, and the loaders only return root asset types in bundles.
	auto checkedLoad = [&]<class T>(const string& key) -> smart_refctd_ptr<T>
	{
		// The `IAssetManager::getAsset` function is very complex, in essencee it:
		// 1. takes a cache key or an IFile, if you gave it an `IFile` skip to step 3
		// 2. it consults the loader override about how to get an `IFile` from your cache key
		// 3. handles any failure in opening an `IFile` (which is why it takes a supposed filename), it allows the override to give a different file
		// 4. tries to derive a working directory if you haven't provided one
		// 5. looks for the assets in the cache if you haven't disabled that in the loader parameters
		// 5a. lets the override choose relevant assets from the ones found under the cache key
		// 5b. if nothing was found it lets the override intervene one last time
		// 6. if there's no file to load from, return no assets
		// 7. try all loaders associated with a file extension
		// 8. then try all loaders by opening the file and checking if it will load
		// 9. insert loaded assets into cache if required
		// 10. restore assets from dummy state if needed (more on that in other examples)
		// Take the docs with a grain of salt, the `getAsset` will be rewritten to deal with restores better in the near future.
		nbl::asset::SAssetBundle bundle = assetManager->getAsset(key,lp);
		if (bundle.getContents().empty())
		{
			logger->log("Asset %s failed to load! Are you sure it exists?",ILogger::ELL_ERROR,key.c_str());
			return nullptr;
		}
		// All assets derive from `nbl::asset::IAsset`, and can be casted down if the type matches
		static_assert(std::is_base_of_v<nbl::asset::IAsset,T>);
		// The type of the root assets in the bundle is not known until runtime, so this is kinda like a `dynamic_cast` which will return nullptr on type mismatch
		auto typedAsset = IAsset::castDown<T>(bundle.getContents()[0]); // just grab the first asset in the bundle
		if (!typedAsset)
			logger->log("Asset type mismatch want %d got %d !",ILogger::ELL_ERROR,T::AssetType,bundle.getAssetType());
		return typedAsset;
	};
	//PNG loader test
	if (auto cpuImage = checkedLoad.operator()<nbl::asset::ICPUImage>("Cerberus_by_Andrew_Maximov/Textures/Cerberus_H.png"))
	{
		// A deep delve into the different asset types is a topic for another example.
		// The important thing to note is that an IImage doesn't care about a lot of things that an IImageView does like Array vs Non-Array typenesss.
		// Also that Views can reinterpret the texel block format as well as subresources (layer, mip-map ranges) of the Image.
		// This is why most Loaders for image-files except for KTX and DDS return Images as opposed to Views because there are missing semantics.
		ICPUImageView::SCreationParams imgViewParams;
		imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		imgViewParams.format = cpuImage->getCreationParameters().format;
		imgViewParams.image = core::smart_refctd_ptr<ICPUImage>(cpuImage);
		imgViewParams.viewType = ICPUImageView::ET_2D;
		imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
		smart_refctd_ptr<nbl::asset::ICPUImageView> imageView = ICPUImageView::create(std::move(imgViewParams));

		// However the writers all take IImageViews because the engine is supposed to know already the OETF, EOTF, mip-chain and exact type.
		nbl::asset::IAssetWriter::SAssetWriteParams wp(imageView.get());
		wp.workingDirectory = CWD;
		assetManager->writeAsset("pngWriteSuccessful.png", wp);
	}
	//JPEG loader test
	if (auto cpuImage = checkedLoad.operator()<nbl::asset::ICPUImage>("dwarf.jpg"))
	{
		ICPUImageView::SCreationParams imgViewParams;
		imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		imgViewParams.format = cpuImage->getCreationParameters().format;
		imgViewParams.image = core::smart_refctd_ptr<ICPUImage>(cpuImage);
		imgViewParams.viewType = ICPUImageView::ET_2D;
		imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
		auto imageView = ICPUImageView::create(std::move(imgViewParams));

		IAssetWriter::SAssetWriteParams wp(imageView.get());
		wp.workingDirectory = CWD;
		assetManager->writeAsset("jpgWriteSuccessful.jpg", wp);
	}
	
	// opening a `.zip` archive and mounting it under a virtual path
	auto bigarch = system->openFileArchive(CWD/"../../media/sponza.zip");
	system->mount(std::move(bigarch),"sponza");
	//TODO OBJ loader test 
	{
		//auto bundle = assetManager->getAsset("../../media/sponza.obj", lp);
		//assert(!bundle.getContents().empty());
		//auto cpumesh = bundle.getContents().begin()[0];
		//auto cpumesh_raw = static_cast<ICPUMesh*>(cpumesh.get());
		//
		//IAssetWriter::SAssetWriteParams wp(cpumesh.get());
		//assetManager->writeAsset("objWriteSuccessful.obj", wp);
	}

	// copying files around
	system->copy(CWD/"pngWriteSuccessful.png", CWD/"pngCopy.png");
	// creating a directory and copying a whole directory
	system->createDirectory(CWD/"textures1");
	system->copy(CWD/"textures1", CWD/"textures");
	// copying from a mounted archive to folder on disk
	system->copy("sponza/textures", CWD/"textures");

	// you can also list items in directories taking into account any virtual mountpoints
	const auto items = system->listItemsInDirectory(CWD/"textures");
	for (const auto& item : items)
		logger->log("%s",system::ILogger::ELL_DEBUG,item.generic_string().c_str());

/* TODO: Tar Archive reader test
	system->moveFileOrDirectory("file.tar","movedFile.tar");
	{
		system::future<smart_refctd_ptr<IFile>> fut;
		system->createFile(fut, "tarArch/file.txt", IFile::ECF_READ);
		auto file = fut.get();
		{
			system::future<smart_refctd_ptr<IFile>> fut;
			system->createFile(fut, "tarArch/file.txt", IFile::ECF_READ);
			file = fut.get();
		}
		std::string str(5, '\0');
		system::future<size_t> readFut;
		file->read(readFut, str.data(), 0, 5);
		readFut.get();
		std::cout << str << std::endl;
	}
*/
	return 0;
}
