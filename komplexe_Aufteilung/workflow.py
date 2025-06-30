from config import *
from Image import loadImage

def returnImage(generatedImage: Any) -> io.BytesIO:
    image = get_value_at_index(generatedImage, 0)
    arr = image.cpu().numpy()

    # Remove extra dimensions if present
    arr = np.squeeze(arr)  # removes dimensions of size 1

    # If shape is (H, W, 4) or (H, W, 3), continue. If (4, H, W), transpose.
    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)

    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)

    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    return img_buffer


def format_text_for_field(first_name, last_name, animal_name, line_length=13, lines=3):
    import textwrap
    text = f"{first_name} {last_name} {animal_name}"
    wrapped = textwrap.wrap(text, width=line_length)
    # Ensure exactly 'lines' lines (pad with empty strings if needed)
    wrapped = wrapped[:lines] + [""] * (lines - len(wrapped))
    return "\n".join(wrapped)


def generateImage(image_bytes: bytes, animal_type: str, first_name: str, last_name: str, animal_name: str) -> io.BytesIO:
    image_rembg_remove_background_62, cliptextencode_44, vaeencode_49 = loadImage(image_bytes)
    depthanythingpreprocessor_36 = depthanythingpreprocessor.execute(
        ckpt_name="depth_anything_vitl14.pth",
        resolution=1024,
        image=get_value_at_index(image_rembg_remove_background_62, 0),
    )

    controlnetapplyadvanced_40 = controlnetapplyadvanced.apply_controlnet(
        strength=0.8500000000000002,
        start_percent=0,
        end_percent=1,
        positive=get_value_at_index(cliptextencode_44, 0),
        negative=get_value_at_index(cliptextencode_66, 0),
        control_net=get_value_at_index(controlnetloader_47, 0),
        image=get_value_at_index(depthanythingpreprocessor_36, 0),
        vae=get_value_at_index(vaeloader_32, 0),
    )

    fluxguidance_43 = fluxguidance.append(
        guidance=7,
        conditioning=get_value_at_index(controlnetapplyadvanced_40, 0),
    )

    fluxguidance_42 = fluxguidance.append(
        guidance=1,
        conditioning=get_value_at_index(controlnetapplyadvanced_40, 1),
    )

    ksampler_3 = ksampler.sample(
        seed=random.randint(1, 2**64),
        steps=20,
        cfg=1,
        sampler_name="euler",
        scheduler="normal",
        denoise=1,
        model=get_value_at_index(checkpointloadersimple_38, 0),
        positive=get_value_at_index(fluxguidance_43, 0),
        negative=get_value_at_index(fluxguidance_42, 0),
        latent_image=get_value_at_index(vaeencode_49, 0),
    )

    vaedecode_8 = vaedecode.decode(
        samples=get_value_at_index(ksampler_3, 0),
        vae=get_value_at_index(vaeloader_32, 0),
    )

    imagecompositemasked_57 = imagecompositemasked.composite(
        x=0,
        y=0,
        resize_source=True,
        destination=get_value_at_index(loadimage_58, 0),
        source=get_value_at_index(vaedecode_8, 0),
        mask=get_value_at_index(loadimage_58, 1),
    )

    textonimage_59 = textonimage.apply_text(
        text=format_text_for_field(first_name, last_name, animal_name), ####### Custom Text #######
        x=853,
        y=898,
        font_size=16,
        text_color="#d3c7b6",
        text_opacity=1,
        use_gradient=False,
        start_color="#ff0000",
        end_color="#0000ff",
        angle=0,
        stroke_width=0,
        stroke_color="#000000",
        stroke_opacity=1,
        shadow_x=0,
        shadow_y=0,
        shadow_color="#000000",
        shadow_opacity=1,
        font_file="en-AllRoundItalic.ttf",
        image=get_value_at_index(imagecompositemasked_57, 0),
    )

    return returnImage(textonimage_59)