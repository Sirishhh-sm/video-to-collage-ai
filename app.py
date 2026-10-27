
import streamlit as st
import cv2
import math
import os
import tempfile
from PIL import Image, ImageDraw
from google import genai
from io import BytesIO

st.set_page_config(
    page_title="Video-to-Collage Generator",
    page_icon="🎬",
    layout="wide"
)


def create_collage(frames):
    thumb_width = 240
    thumb_height = 180
    padding = 8
    label_height = 24
    columns = 3

    rows = math.ceil(len(frames) / columns)

    collage_width = columns * thumb_width + (columns + 1) * padding
    collage_height = rows * (thumb_height + label_height) + (rows + 1) * padding

    collage = Image.new(
        "RGB",
        (collage_width, collage_height),
        "#f2f2f2"
    )

    draw = ImageDraw.Draw(collage)

    for i, (frame, timestamp) in enumerate(frames):
        image = Image.fromarray(frame)

        # Crop images to equal-size cells
        image_ratio = image.width / image.height
        target_ratio = thumb_width / thumb_height

        if image_ratio > target_ratio:
            new_width = int(image.height * target_ratio)
            left = (image.width - new_width) // 2
            image = image.crop(
                (left, 0, left + new_width, image.height)
            )
        else:
            new_height = int(image.width / target_ratio)
            top = (image.height - new_height) // 2
            image = image.crop(
                (0, top, image.width, top + new_height)
            )

        image = image.resize((thumb_width, thumb_height))

        x = padding + (i % columns) * (thumb_width + padding)
        y = padding + (i // columns) * (thumb_height + label_height + padding)

        collage.paste(image, (x, y))
        draw.text(
            (x + 5, y + thumb_height + 4),
            f"{timestamp}s",
            fill="black"
        )

    return collage


st.title("🎬 Video-to-Collage Generator")
st.write(
    "Upload a video, select the image interval, "
    "and create your collage."
)

uploaded_video = st.file_uploader(
    "Upload your video",
    type=["mp4", "mov", "avi", "mkv"]
)

interval = st.number_input(
    "Extract one image every (seconds)",
    min_value=1,
    max_value=60,
    value=5,
    step=1
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Generate Collage", type="primary"):
        temp_path = None
        cap = None

        try:
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_video.getvalue())
                temp_path = temp_file.name

            cap = cv2.VideoCapture(temp_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if not cap.isOpened() or fps <= 0 or total_frames <= 0:
                st.error("Could not read this video. Try another file.")
            else:
                duration = total_frames / fps
                frames = []

                for sec in range(0, math.ceil(duration), int(interval)):
                    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                    success, frame = cap.read()

                    if success:
                        frame = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )
                        frames.append((frame, sec))

                if not frames:
                    st.error("No frames could be extracted.")
                else:
                    collage = create_collage(frames)

                    st.success(
                        f"Collage created! {len(frames)} images extracted."
                    )

                    st.subheader("Your Collage")
                    st.image(collage, use_container_width=True)

                    output = BytesIO()
                    collage.save(output, format="PNG")

                    st.download_button(
                        "Download Collage",
                        data=output.getvalue(),
                        file_name="video_collage.png",
                        mime="image/png"
                    )

        except Exception as error:
            st.error(f"Something went wrong: {error}")

        finally:
            if cap is not None:
                cap.release()

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

# AI Script Generator
if uploaded_video is not None:
    st.divider()
    st.subheader("✨ AI Script Generator")

    script_style = st.selectbox(
        "Choose script style",
        [
            "YouTube narration",
            "Instagram Reel",
            "Educational explanation",
            "Product video",
            "Storytelling"
        ]
    )

    script_language = st.selectbox(
        "Choose script language",
        ["English", "Hindi", "Hinglish"]
    )

    if st.button("Generate AI Script"):
        temp_path = None
        cap = None

        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)

            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_video.getvalue())
                temp_path = temp_file.name

            cap = cv2.VideoCapture(temp_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if not cap.isOpened() or fps <= 0 or total_frames <= 0:
                st.error("Could not read this video.")
            else:
                duration = total_frames / fps
                frames_for_ai = []

                for sec in [0, duration / 4, duration / 2,
                            3 * duration / 4, max(0, duration - 0.1)]:
                    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                    success, frame = cap.read()

                    if success:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                        image.thumbnail((768, 768))
                        frames_for_ai.append(image)

                if not frames_for_ai:
                    st.error("No frames could be extracted.")
                else:
                    prompt = f"""
                    Analyze these frames from a video and create a script.

                    Style: {script_style}
                    Language: {script_language}

                    Include:
                    1. An engaging title
                    2. A short description of the visible scenes
                    3. An opening hook
                    4. A narration script in logical order
                    5. A closing line

                    Do not invent events or facts that are not visible.
                    Make the script natural and easy to speak.
                    """
                    script = None
                    last_error = None

                    models_to_try = [
                        "gemini-3.8-flash",
                        "gemini-3.7-flash",
                        "gemini-3.6-flash",
                    ]

                    with st.spinner("Trying Gemini models..."):
                        for model_name in models_to_try:
                            try:
                                response = client.models.generate_content(
                                    model=model_name,
                                    contents=[prompt, *frames_for_ai]
                                )

                                if response.text:
                                    script = response.text
                                    st.caption(f"Generated using {model_name}")
                                    break

                            except Exception as error:
                                last_error = error

                    if not script and last_error:
                        raise last_error
                              

                    if script:
                        st.success("Your script is ready!")
                        st.markdown(script)

                        st.download_button(
                            "Download Script",
                            data=script,
                            file_name="generated_script.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Gemini returned an empty response.")

        except Exception as error:
            st.error("Script generation failed. Check your API key, model availability, internet connection, and API quota.")
            st.caption(str(error))

        finally:
            if cap is not None:
                cap.release()

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)