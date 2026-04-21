(() => {
	"use strict";

	class Cursor {
		constructor(bytes, offset = 0) {
			this.bytes = bytes;
			this.view = new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
			this.offset = offset;
		}

		readU8() {
			return this.view.getUint8(this.offset++);
		}

		readI32() {
			const value = this.view.getInt32(this.offset,true);
			this.offset += 4;
			return value;
		}

		readU32() {
			const value = this.view.getUint32(this.offset,true);
			this.offset += 4;
			return value;
		}

		readU64() {
			const lo = this.view.getUint32(this.offset,true);
			const hi = this.view.getUint32(this.offset + 4,true);
			this.offset += 8;
			return hi*0x100000000 + lo;
		}

		readString() {
			const start = this.offset;
			while (this.offset < this.bytes.length && this.bytes[this.offset] !== 0)
				++this.offset;
			const value = String.fromCharCode(...this.bytes.subarray(start,this.offset));
			++this.offset;
			return value;
		}

		skip(bytes) {
			this.offset += bytes;
		}
	}

	function sampleByteSize(pixelType) {
		if (pixelType === 1)
			return 2;
		if (pixelType === 2 || pixelType === 0)
			return 4;
		throw new Error("Unsupported EXR pixel type " + pixelType);
	}

	function parseChlist(bytes) {
		const cursor = new Cursor(bytes);
		const channels = [];
		while (cursor.offset < bytes.length && bytes[cursor.offset] !== 0) {
			const name = cursor.readString();
			const pixelType = cursor.readI32();
			cursor.skip(1);
			cursor.skip(3);
			const xSampling = cursor.readI32();
			const ySampling = cursor.readI32();
			if (xSampling !== 1 || ySampling !== 1)
				throw new Error("Subsampled EXR channels are not supported");
			channels.push({
				name,
				component: name.split(".").pop().toUpperCase(),
				pixelType,
				bytes: sampleByteSize(pixelType)
			});
		}
		return channels;
	}

	function parseBox2i(bytes) {
		const cursor = new Cursor(bytes);
		return {
			minX: cursor.readI32(),
			minY: cursor.readI32(),
			maxX: cursor.readI32(),
			maxY: cursor.readI32()
		};
	}

	function parseHeader(bytes) {
		const cursor = new Cursor(bytes);
		if (cursor.readU32() !== 20000630)
			throw new Error("Input is not an OpenEXR file");

		cursor.readU32();
		const attributes = {};
		while (bytes[cursor.offset] !== 0) {
			const name = cursor.readString();
			const type = cursor.readString();
			const size = cursor.readU32();
			const value = bytes.subarray(cursor.offset,cursor.offset + size);
			cursor.skip(size);
			attributes[name] = { type, value };
		}
		cursor.skip(1);

		if (!attributes.channels || !attributes.compression || !attributes.dataWindow)
			throw new Error("EXR header is missing required attributes");

		const channels = parseChlist(attributes.channels.value);
		const dataWindow = parseBox2i(attributes.dataWindow.value);
		const width = dataWindow.maxX - dataWindow.minX + 1;
		const height = dataWindow.maxY - dataWindow.minY + 1;
		const compression = attributes.compression.value[0];
		const linesPerChunk = compression === 3 ? 16 : 1;
		const chunkCount = Math.ceil(height/linesPerChunk);
		const offsets = [];
		for (let i = 0; i < chunkCount; ++i)
			offsets.push(cursor.readU64());

		return {
			channels,
			compression,
			dataWindow,
			width,
			height,
			linesPerChunk,
			offsets,
			bytesPerLine: channels.reduce((sum,channel) => sum + channel.bytes*width,0)
		};
	}

	function halfToFloat(value) {
		const sign = (value & 0x8000) ? -1 : 1;
		const exponent = (value >> 10) & 0x1f;
		const mantissa = value & 0x03ff;
		if (exponent === 0)
			return sign*(mantissa === 0 ? 0 : Math.pow(2,-14)*(mantissa/1024));
		if (exponent === 31)
			return mantissa === 0 ? sign*Infinity : NaN;
		return sign*Math.pow(2,exponent - 15)*(1 + mantissa/1024);
	}

	function readSample(view, offset, pixelType) {
		if (pixelType === 1)
			return halfToFloat(view.getUint16(offset,true));
		if (pixelType === 2)
			return view.getFloat32(offset,true);
		if (pixelType === 0)
			return view.getUint32(offset,true);
		return 0;
	}

	async function inflateZlib(bytes) {
		if (!("DecompressionStream" in window))
			throw new Error("Browser does not expose DecompressionStream for ZIP-compressed EXR chunks");
		const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
		return new Uint8Array(await new Response(stream).arrayBuffer());
	}

	function reconstructZip(bytes) {
		const predicted = new Uint8Array(bytes);
		for (let i = 1; i < predicted.length; ++i)
			predicted[i] = (predicted[i - 1] + predicted[i] - 128) & 0xff;

		const output = new Uint8Array(predicted.length);
		let first = 0;
		let second = Math.floor((predicted.length + 1)/2);
		for (let out = 0; out < output.length; ++out)
			output[out] = (out & 1) === 0 ? predicted[first++] : predicted[second++];
		return output;
	}

	async function decodeChunk(compression, bytes) {
		if (compression === 0)
			return bytes;
		if (compression === 2 || compression === 3)
			return reconstructZip(await inflateZlib(bytes));
		throw new Error("Unsupported EXR compression " + compression);
	}

	async function loadExrBytes(url) {
		const response = await fetch(url);
		if (!response.ok)
			throw new Error("Could not load " + url + " (" + response.status + ")");
		return new Uint8Array(await response.arrayBuffer());
	}

	async function decodeExr(bytes) {
		const header = parseHeader(bytes);
		const sourceView = new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
		const pixels = new Float32Array(header.width*header.height*4);
		for (let pixel = 0; pixel < header.width*header.height; ++pixel)
			pixels[pixel*4 + 3] = 1;

		for (let chunkIndex = 0; chunkIndex < header.offsets.length; ++chunkIndex) {
			const chunkOffset = header.offsets[chunkIndex];
			const y = sourceView.getInt32(chunkOffset,true);
			const dataSize = sourceView.getUint32(chunkOffset + 4,true);
			const encoded = bytes.subarray(chunkOffset + 8,chunkOffset + 8 + dataSize);
			const decoded = await decodeChunk(header.compression,encoded);
			const decodedView = new DataView(decoded.buffer,decoded.byteOffset,decoded.byteLength);
			const rowStart = y - header.dataWindow.minY;
			const rowCount = Math.min(header.linesPerChunk,header.height - rowStart);

			for (let row = 0; row < rowCount; ++row) {
				let channelOffset = row*header.bytesPerLine;
				for (const channel of header.channels) {
					const componentIndex = channel.component === "R" ? 0 : channel.component === "G" ? 1 : channel.component === "B" ? 2 : channel.component === "A" ? 3 : -1;
					for (let x = 0; x < header.width; ++x) {
						if (componentIndex >= 0) {
							const sampleOffset = channelOffset + x*channel.bytes;
							pixels[((rowStart + row)*header.width + x)*4 + componentIndex] = readSample(decodedView,sampleOffset,channel.pixelType);
						}
					}
					channelOffset += header.width*channel.bytes;
				}
			}
		}

		return {
			width: header.width,
			height: header.height,
			channels: header.channels.map((channel) => channel.name),
			pixels
		};
	}

	function toSrgb(value) {
		const clamped = Math.min(1,Math.max(0,Number.isFinite(value) ? value : 0));
		if (clamped <= 0.0031308)
			return Math.round(clamped*12.92*255);
		return Math.round((1.055*Math.pow(clamped,1/2.4) - 0.055)*255);
	}

	function formatChannels(channels) {
		const canonical = ["R","G","B","A"];
		const components = new Set(channels.map((channel) => channel.split(".").pop().toUpperCase()));
		const ordered = canonical.filter((channel) => components.has(channel));
		const extras = channels.filter((channel) => !canonical.includes(channel.split(".").pop().toUpperCase()));
		return ordered.concat(extras).join(", ");
	}

	function formatFloat(value) {
		if (!Number.isFinite(value))
			return String(value);
		const abs = Math.abs(value);
		if (abs === 0)
			return "0";
		if (abs < 0.0001 || abs >= 10000)
			return value.toExponential(3);
		return value.toFixed(abs < 1 ? 5 : 3).replace(/0+$/,"").replace(/\.$/,"");
	}

	function formatPixelCoord(value) {
		return String(value);
	}

	function emptyPixelInfoText() {
		return { coords: "x - y -", channels: "R - G - B - A -" };
	}

	function makeImageData(exr, selection) {
		const outputName = (selection.output.identifier + " " + selection.output.title).toLowerCase();
		const variantName = (selection.variant.identifier + " " + selection.variant.label).toLowerCase();
		const isNormal = outputName.includes("normal");
		const isDifference = variantName.includes("difference");
		let differenceScale = 1;
		if (isDifference) {
			let maxValue = 0;
			for (let i = 0; i < exr.pixels.length; i += 4)
				maxValue = Math.max(maxValue,Math.abs(exr.pixels[i]),Math.abs(exr.pixels[i + 1]),Math.abs(exr.pixels[i + 2]));
			differenceScale = maxValue > 0 && maxValue < 1 ? 1/maxValue : 1;
		}

		const image = new ImageData(exr.width,exr.height);
		for (let pixel = 0; pixel < exr.width*exr.height; ++pixel) {
			const source = pixel*4;
			const destination = source;
			for (let channel = 0; channel < 3; ++channel) {
				let value = exr.pixels[source + channel];
				if (isNormal && value < 0)
					value = value*0.5 + 0.5;
				if (isDifference)
					value *= differenceScale;
				image.data[destination + channel] = toSrgb(value);
			}
			image.data[destination + 3] = 255;
		}
		return image;
	}

	function createViewer(elements) {
		const card = elements.card;
		const title = elements.title;
		const status = elements.status;
		const target = elements.target;
		const sceneSelect = elements.sceneSelect;
		const outputSelect = elements.outputSelect;
		const variantTabs = elements.variantTabs;
		const fitButton = elements.fitButton;
		const oneToOneButton = elements.oneToOneButton;
		const zoomOutButton = elements.zoomOutButton;
		const zoomInButton = elements.zoomInButton;
		const pixelInfo = elements.pixelInfo;
		const closeButton = elements.closeButton;

		let manifest = { scenes: [] };
		let active = {
			sceneIndex: 0,
			output: "",
			variant: "render"
		};
		let previewRequestId = 0;
		let currentPreview = null;
		let dragState = null;
		let lastPixelInfoKey = "";
		const exrCache = new Map();
		const previewCache = new Map();
		const maxCachedPreviews = 12;

		function clampZoom(value) {
			return Math.min(64,Math.max(0.05,value));
		}

		function rootRem() {
			return Number.parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
		}

		function lengthFromPixels(value) {
			return (value/rootRem()).toFixed(4).replace(/\.?0+$/,"") + "rem";
		}

		function applyCanvasTransform() {
			if (!currentPreview || !currentPreview.canvas)
				return;
			currentPreview.canvas.style.transform = "translate(" + lengthFromPixels(currentPreview.offsetX) + "," + lengthFromPixels(currentPreview.offsetY) + ") scale(" + currentPreview.scale + ")";
		}

		function setZoom(scale, anchorX, anchorY) {
			if (!currentPreview)
				return;
			const nextScale = clampZoom(scale);
			const rect = target.getBoundingClientRect();
			const viewportX = Number.isFinite(anchorX) ? anchorX - rect.left : rect.width*0.5;
			const viewportY = Number.isFinite(anchorY) ? anchorY - rect.top : rect.height*0.5;
			const imageX = (viewportX - currentPreview.offsetX)/currentPreview.scale;
			const imageY = (viewportY - currentPreview.offsetY)/currentPreview.scale;
			currentPreview.scale = nextScale;
			currentPreview.offsetX = viewportX - imageX*nextScale;
			currentPreview.offsetY = viewportY - imageY*nextScale;
			applyCanvasTransform();
		}

		function zoomBy(factor, anchorX, anchorY) {
			if (currentPreview)
				setZoom(currentPreview.scale*factor,anchorX,anchorY);
		}

		function fitPreview() {
			if (!currentPreview)
				return;
			const viewportWidth = Math.max(1,target.clientWidth);
			const viewportHeight = Math.max(1,target.clientHeight);
			const scale = clampZoom(Math.min(viewportWidth/currentPreview.exr.width,viewportHeight/currentPreview.exr.height)*0.96);
			currentPreview.scale = scale;
			currentPreview.offsetX = (viewportWidth - currentPreview.exr.width*scale)*0.5;
			currentPreview.offsetY = (viewportHeight - currentPreview.exr.height*scale)*0.5;
			applyCanvasTransform();
		}

		function resetZoom() {
			if (!currentPreview)
				return;
			currentPreview.scale = 1;
			currentPreview.offsetX = (Math.max(1,target.clientWidth) - currentPreview.exr.width)*0.5;
			currentPreview.offsetY = (Math.max(1,target.clientHeight) - currentPreview.exr.height)*0.5;
			applyCanvasTransform();
		}

		function imagePixelFromEvent(event) {
			if (!currentPreview)
				return null;
			const rect = target.getBoundingClientRect();
			const x = Math.floor((event.clientX - rect.left - currentPreview.offsetX)/currentPreview.scale);
			const y = Math.floor((event.clientY - rect.top - currentPreview.offsetY)/currentPreview.scale);
			if (x < 0 || y < 0 || x >= currentPreview.exr.width || y >= currentPreview.exr.height)
				return null;
			return { x, y };
		}

		function setPixelInfoText(info, outside = false) {
			if (!pixelInfo)
				return;
			const normalized = typeof info === "string" ? { coords: info, channels: "" } : info;
			const key = normalized.coords + "\n" + normalized.channels + "\n" + outside;
			pixelInfo.classList.toggle("is-outside",outside);
			if (lastPixelInfoKey === key)
				return;
			pixelInfo.replaceChildren();
			const label = document.createElement("span");
			label.textContent = "Pixel";
			const coords = document.createElement("strong");
			coords.textContent = normalized.coords;
			const channels = document.createElement("strong");
			channels.textContent = normalized.channels;
			pixelInfo.append(label,coords,channels);
			lastPixelInfoKey = key;
		}

		function updatePixelInfo(pixel) {
			if (!currentPreview || !pixel) {
				setPixelInfoText(emptyPixelInfoText(),true);
				return;
			}
			const index = (pixel.y*currentPreview.exr.width + pixel.x)*4;
			const pixels = currentPreview.exr.pixels;
			setPixelInfoText({
				coords: "x " + formatPixelCoord(pixel.x) + " y " + formatPixelCoord(pixel.y),
				channels:
					"R " + formatFloat(pixels[index]) +
					"  G " + formatFloat(pixels[index + 1]) +
					"  B " + formatFloat(pixels[index + 2]) +
					"  A " + formatFloat(pixels[index + 3])
			});
		}

		function previewKey(selection) {
			return selection.variant.image + "\n" + selection.output.identifier + "\n" + selection.variant.identifier;
		}

		function trimPreviewCache() {
			while (previewCache.size > maxCachedPreviews)
				previewCache.delete(previewCache.keys().next().value);
		}

		function cachedExr(url) {
			let entry = exrCache.get(url);
			if (entry) {
				exrCache.delete(url);
				exrCache.set(url,entry);
			} else {
				entry = loadExrBytes(url).then((bytes) => decodeExr(bytes));
				exrCache.set(url,entry);
				entry.catch(() => exrCache.delete(url));
			}
			return entry;
		}

		function cachedPreview(selection) {
			const key = previewKey(selection);
			let entry = previewCache.get(key);
			if (entry) {
				previewCache.delete(key);
				previewCache.set(key,entry);
			} else {
				entry = cachedExr(selection.variant.image).then((exr) => ({
					exr,
					imageData: makeImageData(exr,selection)
				}));
				previewCache.set(key,entry);
				entry.catch(() => previewCache.delete(key));
				trimPreviewCache();
			}
			return entry;
		}

		function warmSiblingVariants(selection) {
			for (const variant of selection.output.variants) {
				if (variant.identifier === selection.variant.identifier)
					continue;
				cachedPreview({
					scene: selection.scene,
					output: selection.output,
					variant
				}).catch(() => {});
			}
		}

		function clearPreview() {
			for (const child of [...target.children])
				child.remove();
			currentPreview = null;
			dragState = null;
			updatePixelInfo(null);
		}

		function drawPreview(preview, selection) {
			const canvas = document.createElement("canvas");
			canvas.width = preview.exr.width;
			canvas.height = preview.exr.height;
			canvas.className = "canvas-preview";
			const context = canvas.getContext("2d");
			if (!context)
				throw new Error("Canvas 2D context is not available");
			context.putImageData(preview.imageData,0,0);
			clearPreview();
			target.appendChild(canvas);
			currentPreview = {
				canvas,
				exr: preview.exr,
				offsetX: 0,
				offsetY: 0,
				scale: 1,
				selection
			};
			fitPreview();
			updatePixelInfo(null);
		}

		function sceneAt(index) {
			return manifest.scenes.find((scene) => scene.index === index) || manifest.scenes[0];
		}

		function outputAt(scene, identifier) {
			if (!scene || !scene.outputs.length)
				return null;
			return scene.outputs.find((output) => output.identifier === identifier) || scene.outputs[0];
		}

		function variantAt(output, identifier) {
			if (!output || !output.variants.length)
				return null;
			return output.variants.find((variant) => variant.identifier === identifier) || output.variants[0];
		}

		function syncSceneSelect() {
			sceneSelect.innerHTML = "";
			for (const scene of manifest.scenes) {
				const option = document.createElement("option");
				option.value = String(scene.index);
				option.textContent = (scene.index + 1) + ". " + scene.title;
				sceneSelect.appendChild(option);
			}
			sceneSelect.value = String(active.sceneIndex);
		}

		function syncOutputSelect(scene) {
			outputSelect.innerHTML = "";
			for (const output of scene.outputs) {
				const option = document.createElement("option");
				option.value = output.identifier;
				option.textContent = output.title;
				outputSelect.appendChild(option);
			}
			outputSelect.value = active.output;
		}

		function syncVariantTabs(output) {
			variantTabs.innerHTML = "";
			for (const variant of output.variants) {
				const button = document.createElement("button");
				button.type = "button";
				button.className = "preview-btn" + (variant.identifier === active.variant ? " is-active" : "");
				button.textContent = variant.label;
				button.dataset.hint = "Switch the preview to the " + variant.label.toLowerCase() + " EXR.";
				button.addEventListener("click",() => {
					active.variant = variant.identifier;
					renderActivePreview();
				});
				variantTabs.appendChild(button);
			}
		}

		function stepSelect(select, direction) {
			if (!select || select.options.length < 2)
				return false;
			const next = Math.max(0,Math.min(select.options.length - 1,select.selectedIndex + direction));
			if (next === select.selectedIndex)
				return false;
			select.selectedIndex = next;
			select.dispatchEvent(new Event("change",{ bubbles: true }));
			return true;
		}

		function installSelectWheel(select) {
			select.addEventListener("wheel",(event) => {
				if (Math.abs(event.deltaY) <= Math.abs(event.deltaX))
					return;
				if (stepSelect(select,event.deltaY > 0 ? 1 : -1))
					event.preventDefault();
			},{ passive: false });
		}

		function stepVariant(direction) {
			const selection = syncControls();
			if (!selection)
				return false;
			const variants = selection.output.variants;
			const index = variants.findIndex((variant) => variant.identifier === active.variant);
			const next = Math.max(0,Math.min(variants.length - 1,index + direction));
			if (next === index)
				return false;
			active.variant = variants[next].identifier;
			renderActivePreview();
			return true;
		}

		function syncControls() {
			const scene = sceneAt(active.sceneIndex);
			if (!scene)
				return null;
			active.sceneIndex = scene.index;
			const output = outputAt(scene,active.output);
			if (!output)
				return null;
			active.output = output.identifier;
			const variant = variantAt(output,active.variant);
			if (!variant)
				return null;
			active.variant = variant.identifier;

			syncSceneSelect();
			syncOutputSelect(scene);
			syncVariantTabs(output);
			return { scene, output, variant };
		}

		function showFallback(message, variant) {
			clearPreview();
			const fallback = document.createElement("div");
			fallback.className = "canvas-preview-fallback";
			const text = document.createElement("p");
			text.textContent = message;
			fallback.appendChild(text);
			if (variant && variant.image) {
				const link = document.createElement("a");
				link.href = variant.image;
				link.textContent = "Open EXR";
				fallback.appendChild(link);
			}
			target.appendChild(fallback);
			status.textContent = message;
			status.style.color = "var(--warn)";
		}

		async function renderActivePreview() {
			const selection = syncControls();
			if (!selection)
				return;

			const requestId = ++previewRequestId;
			title.textContent = selection.scene.title + " / " + selection.output.title + " / " + selection.variant.label;
			clearPreview();
			status.textContent = "Loading and decoding " + selection.variant.image + "...";
			status.style.color = "";

			try {
				const preview = await cachedPreview(selection);
				if (requestId !== previewRequestId)
					return;
				drawPreview(preview,selection);
				warmSiblingVariants(selection);
				status.textContent = "Decoded raw EXR in JavaScript: " + preview.exr.width + "x" + preview.exr.height + ", channels " + formatChannels(preview.exr.channels) + ".";
			} catch (error) {
				if (requestId !== previewRequestId)
					return;
				console.error(error);
				showFallback("EXR preview could not be decoded in JavaScript. Use the direct EXR link.",selection.variant);
			}
		}

		function setManifest(nextManifest) {
			manifest = nextManifest || { scenes: [] };
			clearPreview();
			if (!manifest.scenes.length) {
				status.textContent = "No EXR files are available for preview.";
				status.style.color = "var(--warn)";
				return;
			}
			status.textContent = "Select an EXR preview.";
			status.style.color = "";
		}

		function openPreview(payload) {
			if (!manifest.scenes.length) {
				status.textContent = "No EXR files are available for preview.";
				status.style.color = "var(--warn)";
				return;
			}

			active = {
				sceneIndex: Number.isInteger(payload.sceneIndex) ? payload.sceneIndex : 0,
				output: payload.output || "",
				variant: payload.variant || "render"
			};
			card.classList.add("is-open");
			card.scrollIntoView({ behavior: "smooth", block: "start" });
			renderActivePreview();
		}

		sceneSelect.addEventListener("change",() => {
			active.sceneIndex = Number(sceneSelect.value);
			const scene = sceneAt(active.sceneIndex);
			active.output = scene && scene.outputs[0] ? scene.outputs[0].identifier : "";
			active.variant = "render";
			renderActivePreview();
		});

		outputSelect.addEventListener("change",() => {
			active.output = outputSelect.value;
			active.variant = "render";
			renderActivePreview();
		});

		installSelectWheel(sceneSelect);
		installSelectWheel(outputSelect);

		variantTabs.addEventListener("wheel",(event) => {
			if (Math.abs(event.deltaY) <= Math.abs(event.deltaX))
				return;
			if (stepVariant(event.deltaY > 0 ? 1 : -1))
				event.preventDefault();
		},{ passive: false });

		target.addEventListener("wheel",(event) => {
			if (!currentPreview)
				return;
			event.preventDefault();
			zoomBy(Math.exp(-event.deltaY*0.0015),event.clientX,event.clientY);
			updatePixelInfo(imagePixelFromEvent(event));
		},{ passive: false });

		target.addEventListener("pointerdown",(event) => {
			if (!currentPreview || event.button !== 0)
				return;
			event.preventDefault();
			dragState = {
				pointerId: event.pointerId,
				startX: event.clientX,
				startY: event.clientY,
				startOffsetX: currentPreview.offsetX,
				startOffsetY: currentPreview.offsetY,
				moved: false
			};
			try {
				target.setPointerCapture(event.pointerId);
			} catch (_) {
			}
			target.classList.add("is-panning");
		});

		target.addEventListener("pointermove",(event) => {
			if (dragState && currentPreview) {
				const dx = event.clientX - dragState.startX;
				const dy = event.clientY - dragState.startY;
				if (Math.abs(dx) + Math.abs(dy) > 3)
					dragState.moved = true;
				currentPreview.offsetX = dragState.startOffsetX + dx;
				currentPreview.offsetY = dragState.startOffsetY + dy;
				applyCanvasTransform();
			}
			updatePixelInfo(imagePixelFromEvent(event));
		});

		target.addEventListener("pointerup",(event) => {
			if (!dragState || dragState.pointerId !== event.pointerId)
				return;
			const wasMoved = dragState.moved;
			dragState = null;
			target.classList.remove("is-panning");
			try {
				target.releasePointerCapture(event.pointerId);
			} catch (_) {
			}
			if (!wasMoved)
				updatePixelInfo(imagePixelFromEvent(event));
		});

		target.addEventListener("pointercancel",(event) => {
			if (!dragState || dragState.pointerId !== event.pointerId)
				return;
			dragState = null;
			target.classList.remove("is-panning");
		});

		target.addEventListener("pointerleave",() => {
			if (!dragState)
				updatePixelInfo(null);
		});

		window.addEventListener("resize",() => fitPreview());
		fitButton.addEventListener("click",() => fitPreview());
		oneToOneButton.addEventListener("click",() => resetZoom());
		zoomOutButton.addEventListener("click",() => zoomBy(1/1.25));
		zoomInButton.addEventListener("click",() => zoomBy(1.25));
		closeButton.addEventListener("click",() => card.classList.remove("is-open"));

		return {
			setManifest,
			openPreview
		};
	}

	window.DittExrPreview = {
		createViewer,
		decodeExr
	};
})();
