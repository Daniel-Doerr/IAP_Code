    def was_save_images(self, images, output_path='', filename_prefix="ComfyUI", filename_delimiter='_',
                        extension='png', dpi=96, quality=100, optimize_image="true", lossless_webp="false", prompt=None, extra_pnginfo=None,
                        overwrite_mode='false', filename_number_padding=4, filename_number_start='false',
                        show_history='false', show_history_by_prefix="true", embed_workflow="true",
                        show_previews="true"):

        delimiter = filename_delimiter
        number_padding = filename_number_padding
        lossless_webp = (lossless_webp == "true")
        optimize_image = (optimize_image == "true")

        # Define token system
        tokens = TextTokens()

        original_output = self.output_dir
        # Parse prefix tokens
        filename_prefix = tokens.parseTokens(filename_prefix)

        # Setup output path
        if output_path in [None, '', "none", "."]:
            output_path = self.output_dir
        else:
            output_path = tokens.parseTokens(output_path)
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.output_dir, output_path)
        base_output = os.path.basename(output_path)
        if output_path.endswith("ComfyUI/output") or output_path.endswith(r"ComfyUI\output"):
            base_output = ""

        # Check output destination
        if output_path.strip() != '':
            if not os.path.isabs(output_path):
                output_path = os.path.join(comfy_paths.output_directory, output_path)
            if not os.path.exists(output_path.strip()):
                cstr(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.').warning.print()
                os.makedirs(output_path, exist_ok=True)

        # Find existing counter values
        if filename_number_start == 'true':
            pattern = f"(\\d+){re.escape(delimiter)}{re.escape(filename_prefix)}"
        else:
            pattern = f"{re.escape(filename_prefix)}{re.escape(delimiter)}(\\d+)"
        existing_counters = [
            int(re.search(pattern, filename).group(1))
            for filename in os.listdir(output_path)
            if re.match(pattern, os.path.basename(filename))
        ]
        existing_counters.sort(reverse=True)

        # Set initial counter value
        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        # Set initial counter value
        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        # Set Extension
        file_extension = '.' + extension
        if file_extension not in ALLOWED_EXT:
            cstr(f"The extension `{extension}` is not valid. The valid formats are: {', '.join(sorted(ALLOWED_EXT))}").error.print()
            file_extension = "png"

        results = list()
        output_files = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Delegate metadata/pnginfo
            if extension == 'webp':
                img_exif = img.getexif()
                if embed_workflow == 'true':
                    workflow_metadata = ''
                    prompt_str = ''
                    if prompt is not None:
                        prompt_str = json.dumps(prompt)
                        img_exif[0x010f] = "Prompt:" + prompt_str
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            workflow_metadata += json.dumps(extra_pnginfo[x])
                    img_exif[0x010e] = "Workflow:" + workflow_metadata
                exif_data = img_exif.tobytes()
            else:
                metadata = PngInfo()
                if embed_workflow == 'true':
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                exif_data = metadata

            # Delegate the filename stuffs
            if overwrite_mode == 'prefix_as_filename':
                file = f"{filename_prefix}{file_extension}"
            else:
                if filename_number_start == 'true':
                    file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
                else:
                    file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"
                if os.path.exists(os.path.join(output_path, file)):
                    counter += 1

            # Save the images
            try:
                output_file = os.path.abspath(os.path.join(output_path, file))
                if extension in ["jpg", "jpeg"]:
                    img.save(output_file,
                             quality=quality, optimize=optimize_image, dpi=(dpi, dpi))
                elif extension == 'webp':
                    img.save(output_file,
                             quality=quality, lossless=lossless_webp, exif=exif_data)
                elif extension == 'png':
                    img.save(output_file,
                             pnginfo=exif_data, optimize=optimize_image, dpi=(dpi, dpi))
                elif extension == 'bmp':
                    img.save(output_file)
                elif extension == 'tiff':
                    img.save(output_file,
                             quality=quality, optimize=optimize_image)
                else:
                    img.save(output_file,
                             pnginfo=exif_data, optimize=optimize_image)

                cstr(f"Image file saved to: {output_file}").msg.print()
                output_files.append(output_file)

                if show_history != 'true' and show_previews == 'true':
                    subfolder = self.get_subfolder_path(output_file, original_output)
                    results.append({
                        "filename": file,
                        "subfolder": subfolder,
                        "type": self.type
                    })

                # Update the output image history
                update_history_output_images(output_file)

            except OSError as e:
                cstr(f'Unable to save file to: {output_file}').error.print()
                print(e)
            except Exception as e:
                cstr('Unable to save file due to the to the following error:').error.print()
                print(e)

            if overwrite_mode == 'false':
                counter += 1

        filtered_paths = []
        if show_history == 'true' and show_previews == 'true':
            HDB = WASDatabase(WAS_HISTORY_DATABASE)
            conf = getSuiteConfig()
            if HDB.catExists("History") and HDB.keyExists("History", "Output_Images"):
                history_paths = HDB.get("History", "Output_Images")
            else:
                history_paths = None

            if history_paths:

                for image_path in history_paths:
                    image_subdir = self.get_subfolder_path(image_path, self.output_dir)
                    current_subdir = self.get_subfolder_path(output_file, self.output_dir)
                    if not os.path.exists(image_path):
                        continue
                    if show_history_by_prefix == 'true' and image_subdir != current_subdir:
                        continue
                    if show_history_by_prefix == 'true' and not os.path.basename(image_path).startswith(filename_prefix):
                        continue
                    filtered_paths.append(image_path)

                if conf.__contains__('history_display_limit'):
                    filtered_paths = filtered_paths[-conf['history_display_limit']:]

                filtered_paths.reverse()

        if filtered_paths:
            for image_path in filtered_paths:
                subfolder = self.get_subfolder_path(image_path, self.output_dir)
                image_data = {
                    "filename": os.path.basename(image_path),
                    "subfolder": subfolder,
                    "type": self.type
                }
                results.append(image_data)

        if show_previews == 'true':
            return {"ui": {"images": results, "files": output_files}, "result": (images, output_files,)}
        else:
            return {"ui": {"images": []}, "result": (images, output_files,)}